"""Delivery features per response: acoustic syllable/articulation rate and verbatim fillers.

Steps (run in order; each caches its output and skips finished clips):
  --step wav        extract 16 kHz mono WAV from each RecruitView video with ffmpeg
  --step syllables  acoustic syllable nuclei (de Jong & Wempe 2009) -> speech/articulation rate
  --step fillers    filler / disfluency counts from verbatim transcripts
                    (outputs/text/verbatim/<id>.json, written by transcribe_verbatim.py)
  --step merge      join everything into outputs/text/delivery/delivery_features.parquet

These are delivery (how-it-was-said) features. They are kept separate from the text
model inputs and are reported as observable measurements, not as nervousness or ability.

Requires: ffmpeg on PATH, soundfile, praat-parselmouth.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess

import numpy as np
import pandas as pd

from text_common import OUT_DIR, ROOT, SPLITS, load_splits

VIDEO_DIR = ROOT / "RecruitView"
WAV_DIR = OUT_DIR / "wav"
VERBATIM_DIR = OUT_DIR / "verbatim"
DELIVERY_DIR = OUT_DIR / "delivery"

# Syllable-nuclei parameters (de Jong & Wempe 2009 defaults).
SILENCE_DB = -25.0     # threshold relative to the 99th-percentile intensity
MIN_DIP_DB = 2.0       # required intensity dip between two syllable peaks
MIN_PAUSE_S = 0.3      # silences shorter than this do not split speech
MIN_SOUNDING_S = 0.1   # sounding stretches shorter than this are ignored

FILLER_RE = re.compile(r"^\W*(u+m+|u+h+|e+r+m*|h+m+|m+h*m+|a+h+)\W*$", re.I)
EVENT_RE = re.compile(r"^\[[^\]]*\]\W*$")


def all_responses() -> pd.DataFrame:
    dfs = load_splits()
    return pd.concat([dfs[s] for s in SPLITS], ignore_index=True)


# --------------------------------------------------------------------------- wav

def extract_wavs(df: pd.DataFrame) -> None:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, (rid, fn) in enumerate(zip(df["id"], df["file_name"])):
        out = WAV_DIR / f"{rid}.wav"
        if out.exists() and out.stat().st_size > 0:
            continue
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(VIDEO_DIR / fn),
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)]
        if subprocess.run(cmd, capture_output=True).returncode != 0:
            failed.append(rid)
        if (i + 1) % 200 == 0:
            print(f"wav {i + 1}/{len(df)}", flush=True)
    print(f"wav done; failed: {failed}")


# --------------------------------------------------------------------------- syllables

def _sounding_mask(vals: np.ndarray, threshold: float, dt: float) -> np.ndarray:
    """Frames above threshold, with short silences bridged and short blips removed."""
    on = vals > threshold
    out = on.copy()
    # bridge silences shorter than MIN_PAUSE_S
    i, n = 0, len(on)
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]:
                j += 1
            if 0 < i and j < n and (j - i) * dt < MIN_PAUSE_S:
                out[i:j] = True
            i = j
        else:
            i += 1
    # drop sounding stretches shorter than MIN_SOUNDING_S
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) * dt < MIN_SOUNDING_S:
                out[i:j] = False
            i = j
        else:
            i += 1
    return out


def syllable_nuclei(wav_path) -> dict[str, float]:
    import parselmouth

    snd = parselmouth.Sound(str(wav_path))
    duration = snd.get_total_duration()
    intensity = snd.to_intensity(minimum_pitch=50.0)
    vals = intensity.values[0]
    times = intensity.xs()
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.01
    if len(vals) < 3 or not np.isfinite(vals).any():
        return {"duration_s": duration, "n_syllables": np.nan}

    max99 = float(np.nanpercentile(vals, 99))
    threshold = max(max99 + SILENCE_DB, float(np.nanmin(vals)))
    sounding = _sounding_mask(vals, threshold, dt)
    phonation = float(sounding.sum() * dt)

    # candidate peaks: local maxima above threshold, inside sounding stretches
    peaks = [i for i in range(1, len(vals) - 1)
             if vals[i] > threshold and vals[i] >= vals[i - 1] and vals[i] > vals[i + 1]
             and sounding[i]]
    # keep a peak only if intensity dips by MIN_DIP_DB before the next peak
    kept = []
    for k, p in enumerate(peaks):
        nxt = peaks[k + 1] if k + 1 < len(peaks) else None
        dip = vals[p:nxt + 1].min() if nxt is not None else vals[p:].min()
        if vals[p] - dip > MIN_DIP_DB:
            kept.append(p)
    # keep only voiced peaks
    pitch = snd.to_pitch_ac(time_step=0.02, pitch_floor=30.0, pitch_ceiling=450.0,
                            voicing_threshold=0.25)
    voiced = [p for p in kept if pitch.get_value_at_time(float(times[p])) > 0]
    n = len(voiced)
    return {
        "duration_s": duration,
        "phonation_time_s": phonation,
        "n_syllables": float(n),
        # syllables per second of the whole clip (includes pauses and lead-in silence)
        "speech_rate_syl_s": n / duration if duration > 0 else np.nan,
        # syllables per second of sounding speech only (excludes pauses)
        "articulation_rate_syl_s": n / phonation if phonation > 0 else np.nan,
        "phonation_ratio": phonation / duration if duration > 0 else np.nan,
        "mean_syllable_s": phonation / n if n else np.nan,
    }


def text_syllables(text: str) -> int:
    """Rough orthographic syllable count (vowel groups), used only to sanity-check the
    acoustic detector."""
    total = 0
    for w in re.findall(r"[a-z]+", text.lower()):
        g = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and g > 1 and not w.endswith(("le", "ee")):
            g -= 1
        total += max(g, 1)
    return total


def run_syllables(df: pd.DataFrame) -> None:
    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DELIVERY_DIR / "syllables.parquet"
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["id"])
    rows = done.to_dict("records")
    have = set(done["id"])
    for i, rid in enumerate(df["id"]):
        if rid in have:
            continue
        wav = WAV_DIR / f"{rid}.wav"
        try:
            r = syllable_nuclei(wav) if wav.exists() else {}
        except Exception as e:     # keep going; record the failure
            print(f"{rid}: {e}")
            r = {}
        rows.append({"id": rid, **r})
        if (i + 1) % 200 == 0:
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"syllables {i + 1}/{len(df)}", flush=True)
    res = pd.DataFrame(rows)
    res.to_parquet(out_path, index=False)

    # sanity check: acoustic count should track the transcript's syllable count
    chk = res.merge(df[["id", "answer_text"]], on="id")
    chk["text_syl"] = chk["answer_text"].map(text_syllables)
    ok = chk["n_syllables"].notna() & (chk["text_syl"] > 0)
    r = np.corrcoef(chk.loc[ok, "n_syllables"], chk.loc[ok, "text_syl"])[0, 1]
    ratio = (chk.loc[ok, "n_syllables"] / chk.loc[ok, "text_syl"]).median()
    print(f"syllables done: {int(ok.sum())} clips; corr(acoustic, transcript syllables) = {r:.3f}; "
          f"median acoustic/transcript ratio = {ratio:.2f}")


# --------------------------------------------------------------------------- fillers

def filler_features(words: list[dict]) -> tuple[dict[str, float], list[dict]]:
    # Bracketed tags that are not fillers ([breath], [laughter], ...) are non-speech events:
    # dropped so they are not counted as words.
    toks = [w for w in words if str(w.get("text", "")).strip()
            and (FILLER_RE.match(str(w["text"])) or not EVENT_RE.match(str(w["text"]).strip()))]
    norm = [re.sub(r"[^\w']", "", str(w["text"]).lower()) for w in toks]
    is_filler = [bool(FILLER_RE.match(str(w["text"]))) for w in toks]
    content = [t for t, f in zip(norm, is_filler) if not f and t]
    n_words = len(content)
    n_fill = int(sum(is_filler))
    reps = sum(1 for a, b in zip(content, content[1:]) if a == b)
    partial = sum(1 for w in toks if str(w["text"]).rstrip().endswith("-"))
    starts = [w.get("start") for w in toks if w.get("start") is not None]
    ends = [w.get("end") for w in toks if w.get("end") is not None]
    span = (max(ends) - min(starts)) if starts and ends else np.nan
    evidence = [{"term": str(w["text"]).strip(), "start": w.get("start"), "end": w.get("end")}
                for w, f in zip(toks, is_filler) if f]
    per100 = 100.0 / (n_words + n_fill) if (n_words + n_fill) else np.nan
    return {
        "verbatim_words": float(n_words),
        "filler_count": float(n_fill),
        "filler_per100": n_fill * per100,
        "filler_per_min": 60.0 * n_fill / span if span and span > 0 else np.nan,
        "repetition_per100": reps * per100,
        "partial_word_per100": partial * per100,
    }, evidence


def run_fillers(df: pd.DataFrame) -> None:
    rows, evidence = [], {}
    missing = 0
    for rid in df["id"]:
        p = VERBATIM_DIR / f"{rid}.json"
        if not p.exists():
            missing += 1
            continue
        words = json.loads(p.read_text(encoding="utf-8")).get("words", [])
        feats, ev = filler_features(words)
        rows.append({"id": rid, **feats})
        evidence[rid] = ev
    DELIVERY_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_parquet(DELIVERY_DIR / "fillers.parquet", index=False)
    (DELIVERY_DIR / "filler_evidence.json").write_text(json.dumps(evidence), encoding="utf-8")
    old = df["transcript"].str.lower().str.count(r"\b(?:um+|uh+)\b").sum()
    print(f"fillers done: {len(res)} clips ({missing} without verbatim transcript). "
          f"Total fillers verbatim: {int(res['filler_count'].sum()) if len(res) else 0} "
          f"vs um/uh in original transcripts: {int(old)}")


# --------------------------------------------------------------------------- merge

def merge() -> None:
    parts = [pd.read_parquet(DELIVERY_DIR / f) for f in ("syllables.parquet", "fillers.parquet")
             if (DELIVERY_DIR / f).exists()]
    if not parts:
        raise SystemExit("nothing to merge; run --step syllables and/or --step fillers first")
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="id", how="outer")
    # duration/phonation in seconds are kept for evidence but not as model inputs
    out = out.drop(columns=[c for c in ("duration_s", "phonation_time_s", "n_syllables",
                                        "verbatim_words", "filler_count") if c in out])
    out.to_parquet(DELIVERY_DIR / "delivery_features.parquet", index=False)
    print(f"delivery_features.parquet: {len(out)} rows, columns {list(out.columns[1:])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["wav", "syllables", "fillers", "merge"])
    ap.add_argument("--limit", type=int, default=None, help="first N responses only (smoke test)")
    args = ap.parse_args()
    df = all_responses()
    if args.limit:
        df = df.head(args.limit)
    {"wav": extract_wavs, "syllables": run_syllables, "fillers": run_fillers}.get(
        args.step, lambda _: merge())(df)


if __name__ == "__main__":
    main()
