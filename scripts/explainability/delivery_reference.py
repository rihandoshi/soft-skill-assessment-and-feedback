"""Stage 2 inference library: compare delivery measurements against saved reference distributions.

Everything here works from `results/delivery_descriptors/percentile_tables.json` plus the measurements of
the interview being analysed. It never needs the training participants' data, so it is what runs at
inference on a new interview.

Three layers, kept separate in every output:
  measurement  the Stage 1 value                       {"descriptor": "sentence.speech_rate_wpm", "value": 196}
  comparison   where it sits vs references             {"pop_pct": 94, "speaker_median": 148, "self_delta": +0.32}
  feedback     wording (NOT here; Stage 9 LLM)

Reference frames
  population  empirical quantiles (p0..p100) of TRAIN participants' unique recordings, conditioned on
              duration bucket x question type (fallback to the duration bucket, then pooled, when a cell has
              < min_cell_size recordings). Conciseness descriptors are conditioned on question type only
              (conditioning on duration bucket would be circular).
  speaker     leave-one-out median of the same speaker's OTHER recordings (copies of the same recording
              excluded). For a new user: the other segments of the same interview.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = np.arange(0, 101, dtype=float)
DURATION_BUCKETS = (("short", 0.0, 20.0), ("medium", 20.0, 40.0), ("long", 40.0, float("inf")))


def duration_bucket(audio_duration_s: float) -> str:
    """RecruitView duration buckets are literal lengths: short < 20 s <= medium < 40 s <= long."""
    for name, lo, hi in DURATION_BUCKETS:
        if lo <= audio_duration_s < hi:
            return name
    return "short"


def fit_quantiles(values) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return {"n": int(len(v)), "q": np.percentile(v, LEVELS).round(6).tolist()} if len(v) else {"n": 0, "q": []}


def value_to_pct(q: list[float], value: float) -> float:
    """Percentile of `value` in a stored quantile table (mid-rank for ties, e.g. the many zeros of pause rates)."""
    if value is None or not np.isfinite(value) or not q:
        return float("nan")
    q = np.asarray(q, dtype=float)
    lo, hi = np.searchsorted(q, value, "left"), np.searchsorted(q, value, "right")
    if hi > lo:  # value equals one or more stored quantiles -> centre of that tied run
        return float(LEVELS[lo:hi].mean())
    if lo == 0:
        return 0.0
    if lo >= len(q):
        return 100.0
    x0, x1 = q[lo - 1], q[lo]
    return float(LEVELS[lo - 1] + (value - x0) / (x1 - x0) * (LEVELS[lo] - LEVELS[lo - 1]))


def self_delta(value: float, baseline: float, kind: str) -> float:
    if not (np.isfinite(value) and np.isfinite(baseline)):
        return float("nan")
    if kind == "relative":
        return float((value - baseline) / baseline) if baseline > 0 else float("nan")
    return float(value - baseline)


def speaker_baseline(df: pd.DataFrame, column: str, min_others: int = 2,
                     group_col: str = "participant", recording_col: str = "duplicate_of") -> pd.Series:
    """Leave-one-out median of `column` over the same speaker's other recordings.

    Copies of the same recording (same `recording_col`) are excluded together with the clip itself, otherwise
    a duplicated answer would be compared with itself. NaN if fewer than `min_others` other recordings."""
    out = np.full(len(df), np.nan)
    rec = (df[recording_col] if recording_col in df else pd.Series(np.arange(len(df)), index=df.index)).to_numpy()
    vals = df[column].to_numpy(dtype=float)
    for idx in df.groupby(group_col, sort=False).indices.values():
        recs = rec[idx]
        uniq, first = np.unique(recs, return_index=True)       # one value per distinct recording
        rv = vals[idx][first]
        for j, r in zip(idx, recs):
            others = rv[(uniq != r) & np.isfinite(rv)]
            if len(others) >= min_others:
                out[j] = float(np.median(others))
    return pd.Series(out, index=df.index)


# ============================================================================================
# Reference set
# ============================================================================================
class ReferenceSet:
    def __init__(self, data: dict):
        self.data = data
        self.refs = data["references"]
        self.rules = data["flag_rules"]
        self.min_cell = data["min_cell_size"]

    @classmethod
    def load(cls, path: Path) -> "ReferenceSet":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _chain(self, key: str, bucket: str, qtype: str) -> list[str]:
        cond = self.refs[key]["conditioning"]
        if cond == "bucket_x_qtype":
            return [f"{bucket}|{qtype}", f"{bucket}|*", "*|*"]
        if cond == "qtype":
            return [f"*|{qtype}", "*|*"]
        return ["*|*"]

    def table(self, key: str, bucket: str, qtype: str) -> tuple[str, dict]:
        tables = self.refs[key]["tables"]
        for cell in self._chain(key, bucket, qtype):
            t = tables.get(cell)
            if t and t["n"] >= self.min_cell:
                return cell, t
        return "*|*", tables["*|*"]

    def pct(self, key: str, value: float, bucket: str, qtype: str) -> tuple[float, str, int]:
        cell, t = self.table(key, bucket, qtype)
        return value_to_pct(t["q"], value), cell, t["n"]

    def quantile(self, key: str, level: int, bucket: str, qtype: str) -> float:
        _, t = self.table(key, bucket, qtype)
        return float(t["q"][level]) if t["q"] else float("nan")

    def describe_cell(self, key: str, cell: str, n: int) -> str:
        level = self.refs[key]["level"]
        b, q = cell.split("|")
        unit = {"segment": "answers", "sentence": "sentences", "pause": "mid-sentence pauses"}[level]
        parts = [p for p in (f"{b} answers" if b != "*" else "", f"'{q}' questions" if q != "*" else "") if p]
        return f"train {unit}{' (' + ', '.join(parts) + ')' if parts else ''}, n={n}"


# ============================================================================================
# Per-clip context and comparisons
# ============================================================================================
@dataclass
class ClipContext:
    seg: dict                                   # one row of segment_descriptors (+ bucket)
    sentences: list[dict]                       # sentence_descriptors rows of this clip
    mid_pauses: list[dict]                      # pause_events rows (mid_sentence, counted)
    baseline: dict = field(default_factory=dict)  # descriptor -> speaker leave-one-out median

    @property
    def bucket(self) -> str:
        return self.seg["bucket"]

    @property
    def qtype(self) -> str:
        return self.seg["question_type"]


def compare(ctx: ClipContext, ref: ReferenceSet, key: str, value: float) -> dict:
    """Comparison block for one measurement (population percentile + speaker baseline)."""
    spec = ref.refs[key]
    pct, cell, n = ref.pct(key, value, ctx.bucket, ctx.qtype)
    base = ctx.baseline.get(spec["column"], float("nan")) if spec["level"] == "segment" else float("nan")
    return {"pop_pct": round(pct, 1) if np.isfinite(pct) else None,
            "reference": ref.describe_cell(key, cell, n),
            "speaker_median": round(base, 3) if np.isfinite(base) else None,
            "self_delta": round(self_delta(value, base, spec["delta_kind"]), 3) if np.isfinite(base) else None,
            "self_delta_kind": spec["delta_kind"]}


def _measurement(ref: ReferenceSet, key: str, value: float) -> dict:
    spec = ref.refs[key]
    return {"descriptor": key, "value": round(float(value), 3), "unit": spec["unit"]}


def _magnitude(pct: float, threshold: float, upper: bool = True) -> float:
    """How far past the threshold, on [0, 1] (used only to rank flags)."""
    if not np.isfinite(pct):
        return 0.0
    span = (100 - threshold) if upper else threshold
    return float(np.clip(((pct - threshold) if upper else (threshold - pct)) / max(span, 1e-9), 0, 1))


# ============================================================================================
# Flag rules (thresholds live in the JSON so they can be recalibrated without code changes)
# ============================================================================================
def _flag(kind: str, ref: ReferenceSet, **kw) -> dict:
    rule = ref.rules[kind]
    return {"type": kind, "category": rule["category"], "status": rule["status"],
            "rater_support": rule.get("rater_support"), **kw}


def flag_too_fast(ctx, ref) -> list[dict]:
    r = ref.rules["too_fast"]
    key = "sentence.speech_rate_wpm"
    base = ctx.baseline.get("sent_rate_mean_wpm", float("nan"))
    hits = []
    for s in ctx.sentences:
        v = s.get("speech_rate_wpm")
        if v is None or not np.isfinite(v):
            continue
        pct, cell, n = ref.pct(key, v, ctx.bucket, ctx.qtype)
        delta = self_delta(v, base, "relative")
        ok = pct >= r["pop_pct_min"] and (delta >= r["self_delta_min"] if np.isfinite(delta) else pct >= r["pop_pct_min_no_baseline"])
        if ok:
            hits.append((pct, v, delta, cell, n, s))
    if not hits:
        return []
    pct, v, delta, cell, n, s = max(hits, key=lambda h: h[0])
    return [_flag("too_fast", ref, start_s=s["speech_start"], end_s=s["speech_end"], sentence=s["text"],
                  measurement=_measurement(ref, key, v),
                  comparison={"pop_pct": round(pct, 1), "reference": ref.describe_cell(key, cell, n),
                              "speaker_median": round(base, 1) if np.isfinite(base) else None,
                              "self_delta": round(delta, 3) if np.isfinite(delta) else None, "self_delta_kind": "relative"},
                  n_qualifying_sentences=len(hits), magnitude=_magnitude(pct, r["pop_pct_min"]))]


def flag_too_slow(ctx, ref) -> list[dict]:
    r = ref.rules["too_slow"]
    key = "sentence.speech_rate_wpm"
    base = ctx.baseline.get("sent_rate_mean_wpm", float("nan"))
    hits = []
    for s in ctx.sentences:
        v = s.get("speech_rate_wpm")
        if v is None or not np.isfinite(v):
            continue
        pct, cell, n = ref.pct(key, v, ctx.bucket, ctx.qtype)
        delta = self_delta(v, base, "relative")
        if pct <= r["pop_pct_max"] and np.isfinite(delta) and delta <= r["self_delta_max"]:
            hits.append((pct, v, delta, cell, n, s))
    if not hits:
        return []
    pct, v, delta, cell, n, s = min(hits, key=lambda h: h[0])
    return [_flag("too_slow", ref, start_s=s["speech_start"], end_s=s["speech_end"], sentence=s["text"],
                  measurement=_measurement(ref, key, v),
                  comparison={"pop_pct": round(pct, 1), "reference": ref.describe_cell(key, cell, n),
                              "speaker_median": round(base, 1), "self_delta": round(delta, 3), "self_delta_kind": "relative"},
                  n_qualifying_sentences=len(hits), magnitude=_magnitude(pct, r["pop_pct_max"], upper=False))]


def flag_rushed_opening(ctx, ref) -> list[dict]:
    r = ref.rules["rushed_opening"]
    key = "segment.rushed_opening_ratio"
    v = ctx.seg.get("rushed_opening_ratio")
    if v is None or not np.isfinite(v) or v < r["ratio_min"]:
        return []
    comp = compare(ctx, ref, key, v)
    if comp["pop_pct"] is None or comp["pop_pct"] < r["pop_pct_min"]:
        return []
    first_two = [s for s in ctx.sentences if s["sent_idx"] < 2]
    return [_flag("rushed_opening", ref, start_s=first_two[0]["speech_start"], end_s=first_two[-1]["speech_end"],
                  sentence=" ".join(s["text"] for s in first_two), measurement=_measurement(ref, key, v),
                  comparison=comp, magnitude=_magnitude(comp["pop_pct"], r["pop_pct_min"]))]


def flag_hesitation(ctx, ref) -> list[dict]:
    r = ref.rules["hesitation"]
    if not ctx.mid_pauses:
        return []
    longest = max(ctx.mid_pauses, key=lambda p: p["duration_s"])
    v = longest["duration_s"]
    threshold = max(r["min_s"], ref.quantile("segment.mid_pause_max_s", r["pop_level"], ctx.bucket, ctx.qtype))
    if v < threshold:
        return []
    base = ctx.baseline.get("mid_pause_max_s", float("nan"))
    if np.isfinite(base) and v < base * r["speaker_ratio_min"]:
        return []
    comp = compare(ctx, ref, "segment.mid_pause_max_s", v)
    owner = next((s for s in ctx.sentences if s["sent_idx"] == longest["sent_idx"]), None)
    pause_pct = ref.pct("pause.mid_sentence_duration_s", v, ctx.bucket, ctx.qtype)[0]
    comp["pause_pop_pct"] = round(pause_pct, 1)
    return [_flag("hesitation", ref, start_s=longest["start_s"], end_s=longest["end_s"],
                  sentence=owner["text"] if owner else None,
                  measurement={"descriptor": "pause.mid_sentence_duration_s", "value": round(v, 2), "unit": "s"},
                  comparison=comp, n_long_mid_pauses=sum(p["duration_s"] >= threshold for p in ctx.mid_pauses),
                  magnitude=_magnitude(comp["pop_pct"] or 0, r["pop_level"]))]


def flag_long_answer(ctx, ref) -> list[dict]:
    r = ref.rules["long_answer"]
    key = "segment.speech_span_s"
    v = ctx.seg.get("speech_span_s")
    if v is None or not np.isfinite(v):
        return []
    median = ref.quantile(key, 50, ctx.bucket, ctx.qtype)
    ratio = v / median if median > 0 else float("nan")
    if not (np.isfinite(ratio) and ratio >= r["ratio_to_qtype_median_min"]):
        return []
    comp = compare(ctx, ref, key, v)
    comp["question_type_median_s"] = round(median, 1)
    comp["ratio_to_question_type_median"] = round(ratio, 2)
    return [_flag("long_answer", ref, start_s=ctx.seg.get("response_latency_s"),
                  end_s=(ctx.seg.get("response_latency_s") or 0) + v, sentence=None,
                  measurement=_measurement(ref, key, v), comparison=comp,
                  word_count=ctx.seg.get("word_count"), magnitude=_magnitude(comp["pop_pct"] or 0, 50))]


def flag_slow_start(ctx, ref) -> list[dict]:
    r = ref.rules["slow_start"]
    key = "segment.response_latency_s"
    v = ctx.seg.get("response_latency_s")
    if v is None or not np.isfinite(v) or v < r["min_s"]:
        return []
    comp = compare(ctx, ref, key, v)
    if comp["pop_pct"] is None or comp["pop_pct"] < r["pop_pct_min"]:
        return []
    return [_flag("slow_start", ref, start_s=0.0, end_s=v, sentence=None, measurement=_measurement(ref, key, v),
                  comparison=comp, magnitude=_magnitude(comp["pop_pct"], r["pop_pct_min"]))]


def flag_monotone(ctx, ref) -> list[dict]:
    r = ref.rules["monotone"]
    p = ctx.seg.get("pitch_std_st")
    e = ctx.seg.get("intensity_std_db")
    if p is None or e is None or not (np.isfinite(p) and np.isfinite(e)):
        return []
    cp = compare(ctx, ref, "segment.pitch_std_st", p)
    ce = compare(ctx, ref, "segment.intensity_std_db", e)
    if cp["pop_pct"] is None or ce["pop_pct"] is None:
        return []
    if cp["pop_pct"] > r["pitch_pct_max"] or ce["pop_pct"] > r["energy_pct_max"]:
        return []
    return [_flag("monotone", ref, start_s=None, end_s=None, sentence=None,
                  measurement=_measurement(ref, "segment.pitch_std_st", p), comparison=cp,
                  energy={"measurement": _measurement(ref, "segment.intensity_std_db", e), "comparison": ce},
                  magnitude=_magnitude(cp["pop_pct"], r["pitch_pct_max"], upper=False))]


def flag_trails_off(ctx, ref) -> list[dict]:
    r = ref.rules["trails_off"]
    base = ctx.baseline.get("end_drop_median_db", float("nan"))
    drops = [(s, s["end_drop_db"]) for s in ctx.sentences if s.get("end_drop_db") is not None and np.isfinite(s["end_drop_db"])]
    if len(drops) < r["min_sentences"] or not np.isfinite(base):
        return []
    rel = [(s, d - base) for s, d in drops]
    hit = [(s, x) for s, x in rel if x <= -r["drop_vs_speaker_db"]]
    frac = len(hit) / len(rel)
    if frac < r["sentence_frac_min"]:
        return []
    worst, x = min(hit, key=lambda h: h[1])
    return [_flag("trails_off", ref, start_s=worst["speech_end"] - 0.7, end_s=worst["speech_end"], sentence=worst["text"],
                  measurement={"descriptor": "sentence.end_drop_db", "value": round(worst["end_drop_db"], 2), "unit": "dB"},
                  comparison={"speaker_median": round(base, 2), "self_delta": round(x, 2), "self_delta_kind": "absolute",
                              "fraction_of_sentences": round(frac, 2)},
                  magnitude=float(np.clip(frac, 0, 1)))]


def flag_uptalk(ctx, ref) -> list[dict]:
    r = ref.rules["uptalk"]
    slopes = [(s, s["terminal_slope_st_s"]) for s in ctx.sentences
              if s.get("terminal_slope_st_s") is not None and np.isfinite(s["terminal_slope_st_s"])]
    if len(slopes) < r["min_sentences"]:
        return []
    rising = [(s, v) for s, v in slopes if v >= r["rise_st_s_min"]]
    frac = len(rising) / len(slopes)
    if frac < r["sentence_frac_min"]:
        return []
    worst, v = max(rising, key=lambda h: h[1])
    return [_flag("uptalk", ref, start_s=worst["speech_end"] - 0.3, end_s=worst["speech_end"], sentence=worst["text"],
                  measurement={"descriptor": "sentence.terminal_slope_st_s", "value": round(v, 2), "unit": "st/s"},
                  comparison={"fraction_of_sentences": round(frac, 2)}, magnitude=float(frac))]


# ---- Stage 5: pretrained speech-emotion (SER) flags. Describe how the voice sounded, never an inner state.
def _finite(*vals) -> bool:
    return all(v is not None and np.isfinite(v) for v in vals)


def flag_low_dominance(ctx, ref) -> list[dict]:
    r = ref.rules.get("low_dominance")
    v = ctx.seg.get("ser_dominance_mean")
    key = "self_delta.ser_dominance_mean"
    if r is None or not _finite(v) or key not in ref.refs:
        return []
    cp = compare(ctx, ref, "segment.ser_dominance_mean", v)
    base = ctx.baseline.get("ser_dominance_mean", float("nan"))
    d = self_delta(v, base, ref.refs[key]["delta_kind"])
    if cp["pop_pct"] is None or not np.isfinite(d):
        return []
    self_pct = ref.pct(key, d, ctx.bucket, ctx.qtype)[0]
    if cp["pop_pct"] > r["pop_pct_max"] or not (np.isfinite(self_pct) and self_pct <= r["self_pct_max"]):
        return []
    return [_flag("low_dominance", ref, start_s=ctx.seg.get("ser_dominance_low_start_s"),
                  end_s=ctx.seg.get("ser_dominance_low_end_s"), sentence=None,
                  measurement=_measurement(ref, "segment.ser_dominance_mean", v),
                  comparison={**cp, "speaker_median": round(base, 3), "self_delta": round(d, 3),
                              "self_delta_pct": round(self_pct, 1)},
                  magnitude=_magnitude(cp["pop_pct"], r["pop_pct_max"], upper=False))]


def flag_tension_rise(ctx, ref) -> list[dict]:
    r = ref.rules.get("tension_rise")
    run = ctx.seg.get("ser_tension_run_s")
    if r is None or not _finite(run) or run < r["min_run_s"]:
        return []
    return [_flag("tension_rise", ref, start_s=ctx.seg.get("ser_tension_start_s"), end_s=ctx.seg.get("ser_tension_end_s"),
                  sentence=None, measurement={"descriptor": "segment.ser_tension_run_s", "value": round(run, 1), "unit": "s"},
                  comparison={"frame": "speaker's other answers", "arousal_z_min": 1.0, "pitch_z_min": 1.0},
                  magnitude=float(np.clip(run / 20.0, 0, 1)))]


def flag_composure_drop(ctx, ref) -> list[dict]:
    r = ref.rules.get("composure_drop")
    drop = ctx.seg.get("ser_dominance_drop_sd")
    if r is None or not _finite(drop) or drop <= r["min_drop_sd"]:
        return []
    # The largest drop grows with the number of windows, so it is compared with answers of similar length.
    cp = compare(ctx, ref, "segment.ser_dominance_drop_sd", drop)
    if cp["pop_pct"] is None or cp["pop_pct"] < r["pop_pct_min"]:
        return []
    return [_flag("composure_drop", ref, start_s=ctx.seg.get("ser_drop_start_s"), end_s=ctx.seg.get("ser_drop_end_s"),
                  sentence=None,
                  measurement={"descriptor": "segment.ser_dominance_drop_sd", "value": round(drop, 2), "unit": "speaker SD"},
                  comparison={**cp, "frame": "curve relative to the speaker's other answers"},
                  magnitude=_magnitude(cp["pop_pct"], r["pop_pct_min"]))]


FLAG_FUNCS = {"too_fast": flag_too_fast, "too_slow": flag_too_slow, "rushed_opening": flag_rushed_opening,
              "hesitation": flag_hesitation, "long_answer": flag_long_answer, "slow_start": flag_slow_start,
              "monotone": flag_monotone, "trails_off": flag_trails_off, "uptalk": flag_uptalk,
              "low_dominance": flag_low_dominance, "tension_rise": flag_tension_rise, "composure_drop": flag_composure_drop}


def all_flags(ctx: ClipContext, ref: ReferenceSet) -> list[dict]:
    out = []
    for name, func in FLAG_FUNCS.items():
        out += func(ctx, ref)
    return out


# ============================================================================================
# Scores (dashboard only; feedback is driven by flags)
# ============================================================================================
def signal_score(pct: float, shape: str, band: tuple[float, float]) -> float:
    """0-100. 'upper': more is worse -> 100 x (1 - pct/100). 'upper_band': 100 up to the band's upper edge,
    then linear to 0 at p100. 'band': 100 inside [lo, hi], linear to 0 at p0 / p100."""
    if not np.isfinite(pct):
        return float("nan")
    lo, hi = band
    if shape == "upper":
        return 100.0 * (1 - pct / 100.0)
    if shape == "upper_band":
        return 100.0 if pct <= hi else 100.0 * (100 - pct) / (100 - hi)
    if shape == "band":
        if lo <= pct <= hi:
            return 100.0
        return 100.0 * (pct / lo if pct < lo else (100 - pct) / (100 - hi))
    raise ValueError(shape)


def scores(ctx: ClipContext, ref: ReferenceSet) -> tuple[dict, dict]:
    sig, cat = {}, {}
    for key, spec in ref.refs.items():
        if spec["level"] != "segment" or spec.get("score_shape") in (None, "none") or spec["status"] != "active":
            continue
        v = ctx.seg.get(spec["column"])
        if v is None or not np.isfinite(v):
            continue
        pct = ref.pct(key, v, ctx.bucket, ctx.qtype)[0]
        band = tuple(spec["band"])
        if spec.get("optimal_band_value"):  # Stage 3 EBM band (raw units), only stored when identified
            lo_v, hi_v = spec["optimal_band_value"]
            band = (ref.pct(key, lo_v, ctx.bucket, ctx.qtype)[0], ref.pct(key, hi_v, ctx.bucket, ctx.qtype)[0])
        sig[spec["column"]] = signal_score(pct, spec["score_shape"], band)
    for name, columns in ref.data["categories"].items():
        vals = [sig[c] for c in columns if c in sig and np.isfinite(sig[c])]
        cat[name] = float(np.mean(vals)) if vals else float("nan")
    return sig, cat


def strengths(ctx: ClipContext, ref: ReferenceSet, sig: dict, max_n: int = 2) -> list[dict]:
    """Distinctly favourable signals with rater support: the signal's percentile lies inside its `strength_range`
    (since Stage 3 only answer length, p10-p30 of comparable answers). Ranked lowest percentile first, at most
    one per category. May be empty: an unsupported 'strength' is not invented; the feedback stage takes
    strengths from the other modalities instead."""
    cands = []
    for column in sig:
        key = f"segment.{column}"
        rng = ref.refs[key].get("strength_range")
        v = ctx.seg.get(column)
        if not rng or v is None or not np.isfinite(v):
            continue
        pct = ref.pct(key, v, ctx.bucket, ctx.qtype)[0]
        if np.isfinite(pct):
            distance = max(rng[0] - pct, pct - rng[1], 0.0)  # 0 inside the range
            cands.append((distance, pct, column))
    cands.sort()
    picked, used = [], set()
    for c in cands:  # at most one strength per category, so the second one is not a near-duplicate
        cat = ref.refs[f"segment.{c[2]}"]["category"]
        if c[0] == 0 and cat not in used:
            picked.append(c)
            used.add(cat)
    picked = picked[:max_n]
    out = []
    for distance, pct, column in picked:
        key = f"segment.{column}"
        v = ctx.seg[column]
        out.append({"category": ref.refs[key]["category"], "measurement": _measurement(ref, key, v),
                    "comparison": compare(ctx, ref, key, v), "relative_only": distance > 0})
    return out


# ============================================================================================
# Category profile: one hard class per audio soft-skill category (EXPLAINABILITY_README §3)
# ============================================================================================
# Each category has a fixed set of signals, each oriented toward the category's "high" pole (e.g. pace: fast).
# Category index = mean of the oriented percentiles (0-100) of its signals among comparable train answers
# ("population" frame), or, for speaker-relative signals, among train answers' changes vs the speaker's usual
# ("self_delta" frame). Classes: low / typical / high, cut at the train p15 / p85 of the index. A second class,
# vs_usual (lower / similar / higher), compares the index with the same speaker's other answers.
PROFILE_LEVELS = ("low", "typical", "high")


def oriented_signal_pct(ctx: ClipContext, ref: ReferenceSet, column: str, orient: int, frame: str) -> float:
    v = ctx.seg.get(column)
    if v is None or not np.isfinite(v):
        return float("nan")
    if frame == "population":
        pct = ref.pct(f"segment.{column}", v, ctx.bucket, ctx.qtype)[0]
    else:
        key = f"self_delta.{column}"
        d = self_delta(v, ctx.baseline.get(column, float("nan")), ref.refs[key]["delta_kind"])
        if not np.isfinite(d):
            return float("nan")
        pct = ref.pct(key, d, ctx.bucket, ctx.qtype)[0]
    if not np.isfinite(pct):
        return float("nan")
    return pct if orient > 0 else 100.0 - pct


def category_index(ctx: ClipContext, ref: ReferenceSet, spec: dict) -> tuple[float, list[dict]]:
    parts = []
    for column, orient, frame in spec["signals"]:
        p = oriented_signal_pct(ctx, ref, column, orient, frame)
        parts.append({"descriptor": column, "frame": frame,
                      "toward": spec["labels"]["high"] if orient > 0 else spec["labels"]["low"],
                      "pct_toward_high_pole": None if not np.isfinite(p) else round(p, 1)})
    vals = [p["pct_toward_high_pole"] for p in parts if p["pct_toward_high_pole"] is not None]
    if len(vals) < spec.get("min_signals", 1):
        return float("nan"), parts
    return float(np.mean(vals)), parts


def classify_level(index: float, thresholds: dict) -> str | None:
    if not np.isfinite(index):
        return None
    if index <= thresholds["low_max"]:
        return "low"
    if index >= thresholds["high_min"]:
        return "high"
    return "typical"


def classify_vs_usual(delta: float, thresholds: dict) -> str | None:
    if not np.isfinite(delta):
        return None
    if delta <= thresholds["lower_max"]:
        return "lower"
    if delta >= thresholds["higher_min"]:
        return "higher"
    return "similar"
