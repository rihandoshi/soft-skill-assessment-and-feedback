"""Phase 5: the interview-level layer. What can be said about an interview as a whole, and about change across it.

Pre-registered in results/explainability/stage14_interview_level/preregistration.md. Uses the existing MIT tables only
(CPU, about 1 min):

  aggregates      per interview and dimension: mean index, share of answers at low / high, first two vs last two
                  answers, the strongest and weakest answer
  latency         per answer: first candidate word minus the last interviewer word before it (Stage 7 Whisper words,
                  speakers from the Turker alignment); kept if 0-10 s; interview value = median over >= 3 answers.
                  Test L: against Overall and NotAwkward (expected negative), then also controlling mean hesitation
  trends          last two minus first two answers on the question-adjusted index.
                  T1: is the trend larger than chance reordering of the interview's own answers?
                  T2: does the trend relate to the ratings once the mean is controlled?
  baseline        Ridge from the aggregates to every reliable MIT label, outer folds grouped by student
                  (an audio-only interview baseline for the fusion team; descriptive)
  summary         the `interview_summary` block for all 138 interviews, and interview_cuts.json so that
                  analyze_interview.py can build the same block for a new recording (summarize())

Output  results/explainability/stage14_interview_level/{report.md, interview_cuts.json, interview_summaries.jsonl,
        tables/*.csv}
Run:    python scripts/explainability/interview_summary.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

DIMS = ["hesitation", "expressiveness"]
BETTER = {"hesitation": -1, "expressiveness": +1}          # the direction that counts as "improved"
POINT_AT = {"hesitation": ("most_hesitant", "least_hesitant"), "expressiveness": ("flattest", "liveliest")}
LOW_CUT, HIGH_CUT = 20.0, 80.0
TREND_P, TREND_MIN = 0.10, 15.0                            # T1 rule: exact permutation p and size in index points
LAT_MIN, LAT_MAX, LAT_MIN_ANSWERS = 0.0, 10.0, 3
MIN_TREND_ANSWERS = 4
MIN_RELIABILITY = 0.5
QS = np.linspace(0, 100, 201)

LATENCY_TESTS = [("Overall", -1), ("NotAwkward", -1)]
TREND_TESTS = {"hesitation": [("NoFillers", -1), ("Overall", -1)],
               "expressiveness": [("EngagingTone", +1), ("Excited", +1)]}
LEVEL_SAY = {
    "hesitation": {"high": "hesitated a lot across the interview (filled pauses, stumbles, long pauses)",
                   "low": "spoke with little hesitation across the interview"},
    "expressiveness": {"low": "the voice sounded quite flat throughout the interview",
                       "high": "the delivery was lively throughout the interview"},
}
# Response latency is not a skill of its own: it is hesitation before the answer starts, so it sits under the
# hesitation dimension as `slow_to_start`. Long latency is the tested direction (negative with Overall and
# NotAwkward); a short one is not praised.
LATENCY_SAY = {"high": "often took a while before starting to answer"}
TREND_SAY = {
    ("hesitation", "improved"): "less hesitation in the last two answers than in the first two",
    ("hesitation", "got worse"): "more hesitation in the last two answers than in the first two",
    ("expressiveness", "improved"): "livelier in the last two answers than in the first two",
    ("expressiveness", "got worse"): "flatter in the last two answers than in the first two",
}


# ------------------------------------------------------------------------------------------------ shared pieces
def response_latency(words: list[dict], speaker, answer_start_s: float) -> float:
    """Seconds from the last interviewer word before the answer to the answer's first word. NaN when there is no
    interviewer word before it or the value is outside the pre-registered 0-10 s."""
    ends = [w["end"] for w, s in zip(words, speaker) if s == "interviewer" and w["start"] < answer_start_s]
    if not ends:
        return np.nan
    lat = float(answer_start_s - max(ends))
    return lat if LAT_MIN <= lat <= LAT_MAX else np.nan


def half_change(x: np.ndarray) -> float:
    """Mean of the last two minus mean of the first two (x in time order, len >= 4)."""
    return float(np.mean(x[-2:]) - np.mean(x[:2]))


def perm_changes(x: np.ndarray) -> np.ndarray:
    """half_change under every ordering of x (exact permutation distribution)."""
    return np.array([half_change(x[list(p)]) for p in itertools.permutations(range(len(x)))])


def trend_test(x: np.ndarray) -> tuple[float, float]:
    """(change, two-sided exact permutation p)."""
    c = half_change(x)
    d = perm_changes(x)
    return c, float(np.mean(np.abs(d) >= abs(c) - 1e-9))


def level_of(pct: float) -> str:
    if not np.isfinite(pct):
        return "missing"
    return "low" if pct <= LOW_CUT else "high" if pct >= HIGH_CUT else "normal"


def pct_from_quantiles(q: list[float], x: float) -> float:
    if x is None or not np.isfinite(x):
        return np.nan
    q = np.asarray(q, float)
    lo, hi = np.searchsorted(q, x, "left"), np.searchsorted(q, x, "right")
    if hi > lo:
        return float((QS[lo] + QS[hi - 1]) / 2)
    return float(np.interp(x, q, QS))


def load_cuts(setting: str) -> dict:
    c = json.loads(P.INTERVIEW_CUTS.read_text())
    return {"use": c["use"], "question_means": c["question_means"].get(setting, {}), **c["settings"][setting]}


def summarize(t: pd.DataFrame, setting: str, use: dict, question_means: dict,
              pct_of: Callable[[str, float], float]) -> dict:
    """The interview_summary block. t: one row per answer in time order with question_id, <dim>_index,
    <dim>_level and (optional) latency_s. pct_of(name, value) gives the interview percentile of a mean
    (name = a dimension or 'latency_s')."""
    out = {"setting": setting, "n_answers": int(len(t)), "dimensions": {}, "internal": {}}
    for dim in DIMS:
        x = t[f"{dim}_index"].to_numpy(float)
        ok = np.isfinite(x)
        d: dict = {"feedback_at": "interview" if dim == "expressiveness" else "answer"}
        counts = t[f"{dim}_level"].fillna("missing").value_counts()
        d["answer_levels"] = {k: int(counts.get(k, 0)) for k in ("low", "normal", "high", "missing")}
        mean = float(np.mean(x[ok])) if ok.any() else np.nan
        p = pct_of(dim, mean)
        d["level"] = level_of(p)
        d["say"] = LEVEL_SAY[dim].get(d["level"])
        if ok.sum() >= 2:
            qa = t.loc[ok, "question_id"].to_numpy()
            worst, best = (np.argmax, np.argmin) if dim == "hesitation" else (np.argmin, np.argmax)
            xs = x[ok]
            d["point_at"] = {POINT_AT[dim][0]: str(qa[worst(xs)]), POINT_AT[dim][1]: str(qa[best(xs)])}
        trend_use = use.get(f"{dim}_trend", "not_used")
        d["trend_use"] = trend_use
        d["trend"] = None
        change = pval = np.nan
        if ok.sum() >= MIN_TREND_ANSWERS:
            adj = x[ok] - np.array([question_means.get(dim, {}).get(str(q), 0.0)
                                    for q in t.loc[ok, "question_id"]], float)
            change, pval = trend_test(adj)
            if trend_use != "not_used":
                has = pval <= TREND_P and abs(change) >= TREND_MIN
                d["trend"] = ("improved" if np.sign(change) == BETTER[dim] else "got worse") if has else "steady"
                if has:
                    d["trend_say"] = TREND_SAY[(dim, d["trend"])]
                    q = t.loc[ok, "question_id"].astype(str).tolist()
                    d["trend_answers"] = {"first": q[:2], "last": q[-2:]}
        out["dimensions"][dim] = d
        out["internal"][dim] = {"mean_index": mean, "interview_pct": p, "change": change, "change_p": pval}
    if "latency_s" in t:
        lat = t["latency_s"].to_numpy(float)
        med = float(np.median(lat[np.isfinite(lat)])) if np.isfinite(lat).sum() >= LAT_MIN_ANSWERS else np.nan
        out["internal"]["latency_median_s"] = med
        if use.get("latency"):
            lv = level_of(pct_of("latency_s", med))   # part of hesitation: hesitating before the answer starts
            out["dimensions"]["hesitation"]["slow_to_start"] = {"level": lv, "say": LATENCY_SAY.get(lv)}
    if "latency_median_s" in out["internal"] and not use.get("latency"):
        out["internal"]["latency_note"] = "not used in feedback (Phase 5, test L)"
    return out


# ------------------------------------------------------------------------------------------------ MIT tables
def mit_latency() -> pd.Series:
    import mit_segment_answers as M
    turns_all = M.load_turns()
    ans = pd.read_csv(P.S7_TABLES / "mit_answers.csv")
    out = {}
    for iid in ans["interview"].unique():
        words = json.loads((P.CACHE_MIT / "asr" / f"{iid}_{M.ASR_MODEL}.json").read_text(encoding="utf-8"))
        turns = turns_all[iid]
        speaker, _, per_turn = M.align(turns, words)
        qs = M.find_questions(turns)
        order = [q for q, _ in M.QUESTIONS if q in qs]
        for k, q in enumerate(order):
            span = M.answer_span(turns, per_turn, words, qs[q], qs[order[k + 1]] if k + 1 < len(order) else None)
            if span is not None:
                out[f"MIT_{iid}_{q}"] = response_latency(words, speaker, span["start_s"])
    return pd.Series(out, name="latency_s")


def answer_table() -> pd.DataFrame:
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str}, low_memory=False)
    lv = lv[lv["dataset"] == "mit"]
    ans = pd.read_csv(P.S7_TABLES / "mit_answers.csv")[["id", "interview", "participant", "session", "question",
                                                         "clip_start_s"]]
    seg = pd.read_csv(P.S7_TABLES / "mit_segments.csv")[["id", "audio_duration_s", "quality_composite"]]
    cols = ["id", "speech_span_s", "speaker_f0_hz", *[f"{d}_{k}" for d in DIMS for k in ("index", "level")]]
    a = ans.merge(lv[cols], on="id", how="left").merge(seg, on="id", how="left")
    a = a.join(mit_latency(), on="id")
    a = a.rename(columns={"question": "question_id", "participant": "person"})
    return a.sort_values(["interview", "clip_start_s"]).reset_index(drop=True)


def question_means(a: pd.DataFrame) -> dict:
    return {d: a.groupby("question_id")[f"{d}_index"].mean().round(3).to_dict() for d in DIMS}


def interview_aggregates(a: pd.DataFrame, qm: dict) -> pd.DataFrame:
    rows = []
    for iid, g in a.groupby("interview", sort=False):
        r = {"interview": iid, "person": g["person"].iloc[0], "session": g["session"].iloc[0], "n_answers": len(g),
             "audio_duration_s": g["audio_duration_s"].mean(), "speech_span_s": g["speech_span_s"].mean(),
             "quality_composite": g["quality_composite"].mean(), "speaker_f0_hz": g["speaker_f0_hz"].mean()}
        lat = g["latency_s"].dropna()
        r["latency_s"] = lat.median() if len(lat) >= LAT_MIN_ANSWERS else np.nan
        r["latency_n"] = len(lat)
        for d in DIMS:
            ok = g[g[f"{d}_index"].notna()]
            x = ok[f"{d}_index"].to_numpy(float)
            r[f"{d}_mean"] = x.mean() if len(x) else np.nan
            for lvl in ("low", "high"):
                r[f"{d}_share_{lvl}"] = (ok[f"{d}_level"] == lvl).mean() if len(ok) else np.nan
            r[f"{d}_first2"] = x[:2].mean() if len(x) >= MIN_TREND_ANSWERS else np.nan
            r[f"{d}_last2"] = x[-2:].mean() if len(x) >= MIN_TREND_ANSWERS else np.nan
            if len(x) >= 2:
                r[f"{d}_max_q"] = ok["question_id"].iloc[int(np.argmax(x))]
                r[f"{d}_min_q"] = ok["question_id"].iloc[int(np.argmin(x))]
            if len(x) >= MIN_TREND_ANSWERS:
                adj = x - ok["question_id"].map(qm[d]).to_numpy(float)
                r[f"{d}_trend"], r[f"{d}_trend_p"] = trend_test(adj)
            else:
                r[f"{d}_trend"] = r[f"{d}_trend_p"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def with_labels(iv: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.Series]:
    import mit_label_reliability as LR
    s = LR.load_scores()
    lab = s[s["Worker"] != "AGGR"].groupby("Participant")[LR.LABELS].mean()
    rel = pd.read_csv(P.S7_RELIABILITY).set_index("label")["icc2_k"]
    reliable = [l for l in LR.LABELS if rel.get(l, 0) >= MIN_RELIABILITY]
    return iv.join(lab, on="interview"), reliable, rel


# ------------------------------------------------------------------------------------------------ tests
def t1_null(a: pd.DataFrame, qm: dict, dim: str, n_null: int, rng) -> tuple[float, np.ndarray]:
    """Observed share of interviews with a trend, and the shares when each interview's answer order is shuffled."""
    dists, obs = [], []
    for _, g in a.groupby("interview", sort=False):
        ok = g[g[f"{dim}_index"].notna()]
        if len(ok) < MIN_TREND_ANSWERS:
            continue
        adj = ok[f"{dim}_index"].to_numpy(float) - ok["question_id"].map(qm[dim]).to_numpy(float)
        d = perm_changes(adj)
        absd = np.sort(np.abs(d))
        p_of = lambda c: 1.0 - np.searchsorted(absd, abs(c) - 1e-9, "left") / len(absd)  # noqa: E731
        flag = np.array([(p_of(c) <= TREND_P) and abs(c) >= TREND_MIN for c in d])   # per ordering
        dists.append(flag)
        c = half_change(adj)
        obs.append(p_of(c) <= TREND_P and abs(c) >= TREND_MIN)
    null = np.array([np.mean([f[rng.integers(len(f))] for f in dists]) for _ in range(n_null)])
    return float(np.mean(obs)), null


def ridge_baseline(iv: pd.DataFrame, labels: list[str], n_boot: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    feats = [f"{d}_{k}" for d in DIMS for k in ("mean", "share_low", "share_high", "trend")] + \
        ["latency_s", "speech_span_s"]
    alphas = [0.1, 1, 10, 100, 1000]
    groups = iv["person"].to_numpy()
    oof = pd.DataFrame({"interview": iv["interview"], "person": iv["person"], "session": iv["session"]})
    rows = []

    def fit_predict(Xtr, ytr, Xte, alpha):
        med = np.nanmedian(Xtr, axis=0)
        f = lambda X: np.where(np.isfinite(X), X, med)  # noqa: E731
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(f(Xtr), ytr).predict(f(Xte))

    X = iv[feats].to_numpy(float)
    for lab in labels:
        m = iv[lab].notna().to_numpy()
        Xl, yl, gl = X[m], iv.loc[m, lab].to_numpy(float), groups[m]
        pred = np.full(len(yl), np.nan)
        for tr, te in GroupKFold(5).split(Xl, yl, gl):
            inner = {}
            for al in alphas:
                p_in = np.full(len(tr), np.nan)
                for itr, ite in GroupKFold(5).split(Xl[tr], yl[tr], gl[tr]):
                    p_in[ite] = fit_predict(Xl[tr][itr], yl[tr][itr], Xl[tr][ite], al)
                inner[al] = spearmanr(p_in, yl[tr])[0]
            pred[te] = fit_predict(Xl[tr], yl[tr], Xl[te], max(inner, key=inner.get))
        col = np.full(len(iv), np.nan)
        col[m] = pred
        oof[f"pred_{lab}"] = col
        rho = spearmanr(pred, yl)[0]
        persons = np.unique(gl)
        idx = {p: np.flatnonzero(gl == p) for p in persons}
        boots = []
        for _ in range(n_boot):
            ii = np.concatenate([idx[p] for p in rng.choice(persons, len(persons))])
            boots.append(spearmanr(pred[ii], yl[ii])[0])
        single = {f: spearmanr(iv.loc[m, f], yl, nan_policy="omit")[0] for f in feats}
        best = max(single, key=lambda k: abs(single[k]) if np.isfinite(single[k]) else -1)
        rows.append({"label": lab, "n": int(m.sum()), "ridge_oof_spearman": rho,
                     "ci_lo": np.nanpercentile(boots, 2.5), "ci_hi": np.nanpercentile(boots, 97.5),
                     "best_single_feature": best, "best_single_spearman": single[best]})
    return pd.DataFrame(rows), oof


def md_table(df: pd.DataFrame, fmt: dict | None = None) -> list[str]:
    fmt = fmt or {}
    out = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, r):
            if isinstance(v, float):
                cells.append("" if not np.isfinite(v) else (fmt.get(c, "{:+.2f}").format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from validate_categories_mit import boot_ci, corr_pair, verdict
    rng = np.random.default_rng(args.seed)
    P.S14_TABLES.mkdir(parents=True, exist_ok=True)

    a = answer_table()
    qm = question_means(a)
    iv = interview_aggregates(a, qm)
    ivl, reliable, rel = with_labels(iv)
    ivl_t = ivl.dropna(subset=["Overall"]).reset_index(drop=True)   # interviews with ratings
    print(f"{len(a)} answers, {len(iv)} interviews ({len(ivl_t)} rated), {iv['person'].nunique()} students; "
          f"latency for {a['latency_s'].notna().sum()} answers, {iv['latency_s'].notna().sum()} interviews", flush=True)
    base_ctl = ["audio_duration_s", "quality_composite"]
    ctl_dim = {"hesitation": base_ctl, "expressiveness": base_ctl + ["speaker_f0_hz"]}

    # ---- Test L: latency
    lat_rows = []
    for lab, sign in LATENCY_TESTS:
        r, rp = corr_pair(ivl_t, "latency_s", lab, base_ctl)
        lo, hi, plo, phi = boot_ci(ivl_t, "latency_s", lab, base_ctl, args.n_boot, rng)
        v = verdict(sign, lo, hi, plo, phi, True)
        ctl_h = base_ctl + ["hesitation_mean"]
        _, rph = corr_pair(ivl_t, "latency_s", lab, ctl_h)
        _, _, hlo, hhi = boot_ci(ivl_t, "latency_s", lab, ctl_h, args.n_boot, rng)
        beyond = (hhi < 0) if sign < 0 else (hlo > 0)
        lat_rows.append({"target": lab, "expected": "-" if sign < 0 else "+", "rho": r, "ci_lo": lo, "ci_hi": hi,
                         "partial": rp, "p_lo": plo, "p_hi": phi, "verdict": v,
                         "partial_with_hesitation": rph, "ph_lo": hlo, "ph_hi": hhi,
                         "beyond_hesitation": bool(beyond)})
        print(f"latency vs {lab}: {r:+.2f} [{lo:+.2f},{hi:+.2f}] partial {rp:+.2f} [{plo:+.2f},{phi:+.2f}] {v}; "
              f"+hesitation {rph:+.2f} [{hlo:+.2f},{hhi:+.2f}]", flush=True)
    lat = pd.DataFrame(lat_rows)
    latency_used = bool(((lat["verdict"] == "supported") & lat["beyond_hesitation"]).any())
    latency_disc = {m: corr_pair(ivl_t, "latency_s", m, base_ctl)[1]
                    for m in ["hesitation_mean", "expressiveness_mean", "speech_span_s"]}

    # ---- T1 and T2: trends
    t1_rows, t2_rows, use = [], [], {"latency": latency_used}
    for dim in DIMS:
        obs, null = t1_null(a, qm, dim, args.n_null, rng)
        t1_pass = bool(obs > np.percentile(null, 95))
        t1_rows.append({"dimension": dim, "n_interviews": int(iv[f"{dim}_trend"].notna().sum()),
                        "observed_share": obs, "null_median": float(np.median(null)),
                        "null_p95": float(np.percentile(null, 95)), "passes": t1_pass})
        print(f"T1 {dim}: share with a trend {obs:.3f}, null median {np.median(null):.3f}, "
              f"p95 {np.percentile(null, 95):.3f} -> {'pass' if t1_pass else 'fail'}", flush=True)
        t2_pass = False
        for lab, sign in TREND_TESTS[dim]:
            ctl = ctl_dim[dim] + [f"{dim}_mean"]
            _, rp = corr_pair(ivl_t, f"{dim}_trend", lab, ctl)
            r_raw, _ = corr_pair(ivl_t, f"{dim}_trend", lab, base_ctl)
            _, _, plo, phi = boot_ci(ivl_t, f"{dim}_trend", lab, ctl, args.n_boot, rng)
            ok = (phi < 0) if sign < 0 else (plo > 0)
            wrong = (plo > 0) if sign < 0 else (phi < 0)
            t2_pass |= bool(ok)
            t2_rows.append({"dimension": dim, "target": lab, "expected": "-" if sign < 0 else "+", "rho_raw": r_raw,
                            "partial_given_mean": rp, "p_lo": plo, "p_hi": phi,
                            "verdict": "supported" if ok else "contradicted" if wrong else "no support"})
            print(f"T2 {dim} trend vs {lab}: raw {r_raw:+.2f}, partial given mean {rp:+.2f} [{plo:+.2f},{phi:+.2f}]",
                  flush=True)
        use[f"{dim}_trend"] = "not_used" if not t1_pass else "feedback" if t2_pass else "description"
    t1, t2 = pd.DataFrame(t1_rows), pd.DataFrame(t2_rows)
    print("use:", use, flush=True)

    # ---- Ridge baseline (descriptive)
    base, oof = ridge_baseline(ivl_t, reliable, args.n_boot, rng)
    print(base.round(2).to_string(index=False), flush=True)

    # ---- interview cuts for new recordings
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str}, low_memory=False)
    rv = lv[(lv["dataset"] == "recruitview") & (lv["split"] == "train")]
    if "hesitation_withheld" in rv:   # the recorded-answer gate: withheld levels do not count
        for d in DIMS:
            rv.loc[rv[f"{d}_withheld"].astype(str) == "True", f"{d}_index"] = np.nan
    rv_means = rv.groupby("participant")[[f"{d}_index" for d in DIMS]].mean()
    cuts = {"note": "Quantiles (0, 0.5, ..., 100th) of interview means of the answer indices. live_interview: the "
                    "138 MIT interviews; recorded_answer: RecruitView train speakers' per-person means. Built by "
                    "interview_summary.py (Phase 5).",
            "use": use,
            "question_means": {"live_interview": qm},
            "settings": {
                "live_interview": {"quantiles": {**{d: np.nanpercentile(iv[f"{d}_mean"], QS).tolist() for d in DIMS},
                                                 "latency_s": np.nanpercentile(iv["latency_s"], QS).tolist()},
                                   "n_reference": int(len(iv))},
                "recorded_answer": {"quantiles": {d: np.nanpercentile(rv_means[f"{d}_index"], QS).tolist()
                                                  for d in DIMS},
                                    "n_reference": int(len(rv_means))}}}
    P.INTERVIEW_CUTS.write_text(json.dumps(cuts, indent=1), encoding="utf-8")

    # ---- summaries for all MIT interviews (leave-one-student-out percentiles of the interview mean)
    from fit_answer_levels import pct
    summaries, sum_rows = [], []
    for iid, g in a.groupby("interview", sort=False):
        person = g["person"].iloc[0]
        others = iv[iv["person"] != person]

        def pct_of(name: str, value: float, others=others) -> float:
            col = "latency_s" if name == "latency_s" else f"{name}_mean"
            return float(pct(others[col].to_numpy(float), pd.Series([value]))[0])

        s = summarize(g.reset_index(drop=True), "live_interview", use, qm, pct_of)
        summaries.append({"interview_id": iid, "person": person, "session": g["session"].iloc[0],
                          "interview_summary": s})
        sum_rows.append({"interview": iid, **{f"{d}_{k}": s["dimensions"][d].get(k) for d in DIMS
                                             for k in ("level", "trend")},
                         **{f"{d}_interview_pct": s["internal"][d]["interview_pct"] for d in DIMS}})
    with open(P.S14 / "interview_summaries.jsonl", "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False, default=float) + "\n")
    st = pd.DataFrame(sum_rows)

    # ---- tables
    a.to_csv(P.S14_TABLES / "answers.csv", index=False)
    ivl.to_csv(P.S14_TABLES / "interview_aggregates.csv", index=False)
    lat.to_csv(P.S14_TABLES / "test_latency.csv", index=False)
    t1.to_csv(P.S14_TABLES / "test_trend_t1.csv", index=False)
    t2.to_csv(P.S14_TABLES / "test_trend_t2.csv", index=False)
    base.to_csv(P.S14_TABLES / "ridge_baseline.csv", index=False)
    oof.to_csv(P.S14_TABLES / "ridge_baseline_oof.csv", index=False)
    st.to_csv(P.S14_TABLES / "summary_levels.csv", index=False)

    # ---- validation of the interview level against the Stage 8 targets (consistency check, not a new test)
    lvl_check = []
    for d, lab in (("hesitation", "NoFillers"), ("expressiveness", "EngagingTone")):
        m = ivl_t.merge(st[["interview", f"{d}_level"]], on="interview")
        lvl_check.append({"dimension": d, "label": lab,
                          **{f"mean_{k}": m.loc[m[f"{d}_level"] == k, lab].mean() for k in ("low", "normal", "high")},
                          **{f"n_{k}": int((m[f"{d}_level"] == k).sum()) for k in ("low", "normal", "high")}})
    lvl_check = pd.DataFrame(lvl_check)
    lvl_check.to_csv(P.S14_TABLES / "summary_level_check.csv", index=False)

    # ---- report
    L = ["# Stage 14: the interview-level layer (Phase 5)", "",
         f"Generated by `interview_summary.py`. {len(a)} MIT answers, {len(iv)} interviews ({len(ivl_t)} rated), "
         f"{iv['person'].nunique()} students. 9-rater means; bootstrap 95% CIs over students ({args.n_boot} "
         "resamples). Controls: mean answer clip duration and recording quality (plus speaker pitch for "
         "expressiveness). Pre-registration: `preregistration.md`.", "",
         "## Response latency (test L)", "",
         f"Latency was measured for {a['latency_s'].notna().sum()} of {len(a)} answers "
         f"(median {a['latency_s'].median():.2f} s, IQR {a['latency_s'].quantile(.25):.2f}-"
         f"{a['latency_s'].quantile(.75):.2f} s) and {iv['latency_s'].notna().sum()} interviews.", ""]
    L += md_table(lat[["target", "expected", "rho", "ci_lo", "ci_hi", "partial", "p_lo", "p_hi", "verdict",
                       "partial_with_hesitation", "ph_lo", "ph_hi"]])
    L += ["", "Partial correlation of interview latency with the other measures (length and quality controlled): " +
          ", ".join(f"`{k}` {v:+.2f}" for k, v in latency_disc.items()) + ".", "",
          f"**Decision:** latency {'enters the interview summary' if latency_used else 'stays internal (not used in feedback)'}.",
          "", "## Trends", "", "### T1: larger than chance reordering?", ""]
    L += md_table(t1, {"observed_share": "{:.3f}", "null_median": "{:.3f}", "null_p95": "{:.3f}"})
    L += ["", "### T2: related to the ratings beyond the mean?", ""]
    L += md_table(t2)
    L += ["", "**Decision:** " + "; ".join(f"{d} trend: `{use[f'{d}_trend']}`" for d in DIMS) + ".", "",
          "## Interview levels", "",
          "Counts of the overall interview level (leave-one-student-out percentile of the interview mean):", ""]
    for d in DIMS:
        vc = st[f"{d}_level"].value_counts()
        L.append(f"- {d}: " + ", ".join(f"{k} {int(vc.get(k, 0))}" for k in ("low", "normal", "high", "missing")))
        if use[f"{d}_trend"] != "not_used":
            tv = st[f"{d}_trend"].value_counts()
            L.append(f"  - trend words: " + ", ".join(f"{k} {int(v)}" for k, v in tv.items()))
    L += ["", "Mean rating per interview level (consistency check):", ""]
    L += md_table(lvl_check, {c: "{:.2f}" for c in lvl_check.columns if c.startswith("mean_")})
    L += ["", "## Ridge baseline (descriptive)", "",
          "Features: both dimensions' mean, share low, share high and trend, median latency, mean speaking time. "
          "Outer 5-fold GroupKFold by student, alpha by inner grouped CV. `best_single` is chosen in-sample "
          "(optimistic).", ""]
    L += md_table(base[["label", "n", "ridge_oof_spearman", "ci_lo", "ci_hi", "best_single_feature",
                        "best_single_spearman"]])
    P.S14_REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {P.S14}: {len(summaries)} interview summaries", flush=True)


if __name__ == "__main__":
    main()
