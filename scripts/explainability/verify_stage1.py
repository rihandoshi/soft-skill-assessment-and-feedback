"""Stage 1 verification: automated evidence for the Stage 1 completion checklist.

The checklist asks for human listening. This script does everything that can be checked without ears, so
the listeners know what to focus on, and adds helper columns to the listening sheet:

  A. Fastest / slowest sentences: re-time the extreme sentences with an independent ASR (faster-whisper
     word timestamps) and check that they are still extreme. Catches transcript/timestamp artefacts.
  B. Expressive / monotone answers: cross-check Praat F0 with a different algorithm (librosa pYIN).
     Catches pitch-tracking artefacts (octave errors inflate pitch variability).
  C. Final rises / trailing off: are they large enough to hear, and is the tail still speech (above the
     clip's background-noise level) rather than silence that VAD padded?
  D. -300 dB dropouts: count them, and prove they cannot enter any descriptor (injection test).
  E. Missing values: summarise NaN rates on the unique train recordings and the reason for each.

Outputs (results/explainability/stage1_descriptors/):
  verification_report.md          checklist with automated evidence and what remains for listeners
  listening_check/listening_check_annotated.csv   listening sheet + Whisper text/rate, pYIN agreement, audibility numbers
  verification/*.csv              per-check detail tables

Run:  python scripts/explainability/verify_stage1.py [--whisper-model small.en]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_delivery_descriptors as ex  # noqa: E402

ROOT = ex.ROOT
import paths as P  # noqa: E402  (results layout)
DESC = P.S1
VER_DIR = P.S1_VERIFICATION
WHISPER_CACHE = P.CACHE_WHISPER
SEED = 7
N_EXTREME = 25
N_CONTROL = 40


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------
def load() -> dict:
    sent = pd.read_csv(P.S1_SENTENCES, dtype={"id": str, "participant": str})
    seg = pd.read_csv(P.S1_SEGMENTS, dtype={"id": str, "participant": str})
    settings = pd.read_csv(P.PITCH_SETTINGS, dtype={"participant": str}).set_index("participant")
    sheet = pd.read_csv(P.LISTENING / "listening_check.csv", dtype={"id": str})
    clips = ex.load_clips(None).set_index("id")
    return {"sent": sent, "seg": seg, "settings": settings, "sheet": sheet, "clips": clips}


def frames_for(clip_id: str, participant: str, settings: pd.DataFrame) -> dict:
    s = settings.loc[participant]
    path = ex.CACHE_DIR / "pass2" / f"{clip_id}_{s.pitch_floor_hz:.0f}_{s.pitch_ceiling_hz:.0f}_{ex.CACHE_VERSION}.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------
# A. Speech-rate extremes vs Whisper
# --------------------------------------------------------------------------------------------
def whisper_words(clip_ids: list[str], clips: pd.DataFrame, model_name: str) -> dict[str, list[dict]]:
    WHISPER_CACHE.mkdir(parents=True, exist_ok=True)
    out, todo = {}, []
    for cid in clip_ids:
        cache = WHISPER_CACHE / f"{cid}_{model_name}.json"
        if cache.exists():
            out[cid] = json.loads(cache.read_text(encoding="utf-8"))
        else:
            todo.append(cid)
    if todo:
        from faster_whisper import WhisperModel
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
        for k, cid in enumerate(todo, 1):
            import soundfile as sf  # pass samples, not a path: the installed PyAV is too new for faster-whisper's decoder
            audio, sr = sf.read(clips.loc[cid, "audio_path"], dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == 16000, f"{cid}: expected 16 kHz, got {sr}"
            segments, _ = model.transcribe(audio, language="en", word_timestamps=True,
                                           beam_size=5, condition_on_previous_text=False, vad_filter=False)
            words = [{"w": w.word.strip(), "start": float(w.start), "end": float(w.end), "p": float(w.probability)}
                     for s in segments for w in (s.words or [])]
            (WHISPER_CACHE / f"{cid}_{model_name}.json").write_text(json.dumps(words), encoding="utf-8")
            out[cid] = words
            if k % 20 == 0 or k == len(todo):
                print(f"  whisper {k}/{len(todo)}", flush=True)
    return out


def check_rates(d: dict, model_name: str) -> tuple[pd.DataFrame, dict]:
    sent = d["sent"]
    unique = sent[sent["rate_valid"] & ~sent["is_duplicate"]]
    train = unique[unique["split"] == "train"]
    p5, p95 = train["speech_rate_wpm"].quantile([0.05, 0.95])
    fastest = unique.nlargest(N_EXTREME, "speech_rate_wpm").assign(group="fastest")
    slowest = unique.nsmallest(N_EXTREME, "speech_rate_wpm").assign(group="slowest")
    rest = unique.drop(fastest.index.union(slowest.index))
    control = rest.sample(N_CONTROL, random_state=SEED).assign(group="control")
    sample = pd.concat([fastest, slowest, control])
    words = whisper_words(sorted(sample["id"].unique()), d["clips"], model_name)
    rows = []
    for _, r in sample.iterrows():
        ws = [w for w in words[r["id"]] if r["speech_start"] <= 0.5 * (w["start"] + w["end"]) < r["speech_end"]]
        n = len([w for w in ws if any(c.isalnum() for c in w["w"])])
        rows.append({"group": r["group"], "id": r["id"], "sent_idx": r["sent_idx"],
                     "speech_start": r["speech_start"], "speech_end": r["speech_end"], "duration_s": r["duration_s"],
                     "transcript_words": r["n_words"], "whisper_words": n,
                     "transcript_wpm": r["speech_rate_wpm"], "whisper_wpm": 60 * n / r["duration_s"],
                     "transcript_text": r["text"], "whisper_text": " ".join(w["w"] for w in ws)})
    t = pd.DataFrame(rows)
    t["word_ratio"] = t["whisper_words"] / t["transcript_words"]
    t["still_extreme"] = np.where(t["group"] == "fastest", t["whisper_wpm"] > p95,
                                  np.where(t["group"] == "slowest", t["whisper_wpm"] < p5, np.nan))
    ctrl = t[t["group"] == "control"]
    summary = {"train_p5_wpm": p5, "train_p95_wpm": p95,
               "control_spearman": ctrl[["transcript_wpm", "whisper_wpm"]].corr("spearman").iloc[0, 1],
               "control_median_word_ratio": ctrl["word_ratio"].median()}
    for g in ("fastest", "slowest"):
        sub = t[t["group"] == g]
        summary[f"{g}_still_extreme"] = int(sub["still_extreme"].sum())
        summary[f"{g}_median_word_ratio"] = sub["word_ratio"].median()
        summary[f"{g}_median_transcript_wpm"] = sub["transcript_wpm"].median()
        summary[f"{g}_median_whisper_wpm"] = sub["whisper_wpm"].median()
    return t, summary


# --------------------------------------------------------------------------------------------
# B. Pitch cross-check with pYIN
# --------------------------------------------------------------------------------------------
def pyin_track(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    import librosa
    y, sr = librosa.load(audio_path, sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=1024, hop_length=160)
    t = librosa.times_like(f0, sr=sr, hop_length=160)
    return t, np.where(voiced, f0, np.nan)


def _pyin_job(job: tuple[str, str]) -> tuple[str, tuple[np.ndarray, np.ndarray]]:
    cid, path = job
    return cid, pyin_track(path)


def check_pitch(d: dict, workers: int) -> tuple[pd.DataFrame, dict]:
    seg = d["seg"][~d["seg"]["is_duplicate"]]
    eligible = seg[seg["n_monotony_windows"] >= 3]
    expressive = eligible.nlargest(10, "pitch_std_st").assign(group="most expressive")
    monotone = eligible.nlargest(10, "monotony_frac").assign(group="most monotone")
    control = eligible.drop(expressive.index.union(monotone.index)).sample(25, random_state=SEED).assign(group="control")
    sample = pd.concat([expressive, monotone, control]).drop_duplicates("id")
    jobs = [(cid, d["clips"].loc[cid, "audio_path"]) for cid in sample["id"]]
    tracks = ex.run_parallel(_pyin_job, jobs, workers, "pyin")
    rows = []
    for _, r in sample.iterrows():
        fr = frames_for(r["id"], r["participant"], d["settings"])
        pt, f0 = fr["pitch_t"].astype(float), fr["f0"].astype(float)
        yt, yf = tracks[r["id"]]
        yf_at = np.interp(pt, yt, np.nan_to_num(yf, nan=0.0))
        yv_at = np.interp(pt, yt, np.isfinite(yf).astype(float)) > 0.5
        both = (f0 > 0) & yv_at & (yf_at > 0)
        diff = 12 * np.log2(f0[both] / yf_at[both]) if both.any() else np.array([])
        py_st = 12 * np.log2(yf[np.isfinite(yf)] / np.nanmedian(yf)) if np.isfinite(yf).sum() > 30 else np.array([])
        rows.append({"group": r["group"], "id": r["id"], "participant": r["participant"],
                     "praat_pitch_std_st": r["pitch_std_st"], "pyin_pitch_std_st": float(np.std(py_st)) if len(py_st) else np.nan,
                     "frames_both_voiced": int(both.sum()),
                     "octave_disagreement_frac": float((np.abs(diff) > 6).mean()) if len(diff) else np.nan,
                     "median_abs_diff_st": float(np.median(np.abs(diff))) if len(diff) else np.nan,
                     "monotony_frac": r["monotony_frac"]})
    t = pd.DataFrame(rows)
    summary = {"spearman_praat_vs_pyin_std": t[["praat_pitch_std_st", "pyin_pitch_std_st"]].corr("spearman").iloc[0, 1]}
    for g, sub in t.groupby("group"):
        summary[f"{g}: median octave disagreement"] = sub["octave_disagreement_frac"].median()
        summary[f"{g}: max octave disagreement"] = sub["octave_disagreement_frac"].max()
        summary[f"{g}: median Praat std"] = sub["praat_pitch_std_st"].median()
        summary[f"{g}: median pYIN std"] = sub["pyin_pitch_std_st"].median()
    return t, summary


# --------------------------------------------------------------------------------------------
# C. Audibility of trailing off and final rises
# --------------------------------------------------------------------------------------------
def check_audibility(d: dict, tracks_cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    sent = d["sent"][~d["sent"]["is_duplicate"]]
    trail = sent.nsmallest(N_EXTREME, "end_drop_db").assign(group="strongest trailing off")
    rise = sent.nlargest(N_EXTREME, "terminal_slope_st_s").assign(group="strongest final rise")
    rows = []
    vad, _ = ex.load_vad(set(trail["id"]) | set(rise["id"]))
    for _, r in pd.concat([trail, rise]).iterrows():
        fr = frames_for(r["id"], r["participant"], d["settings"])
        it, db = fr["int_t"].astype(float), fr["int_db"].astype(float)
        ok = db > ex.DROPOUT_DB
        speech = ex._in_regions(it, vad[r["id"]])
        noise = db[ok & ~speech]
        noise_floor = float(np.percentile(noise, 10)) if len(noise) >= 20 else np.nan
        in_sent = (it >= r["speech_start"]) & (it <= r["speech_end"]) & speech & ok
        tail = in_sent & (it >= r["speech_end"] - ex.TRAIL_WINDOW_S)
        sent_peak = float(np.percentile(db[in_sent], ex.TRAIL_PEAK_PCT)) if in_sent.sum() >= 20 else np.nan
        tail_peak = float(np.percentile(db[tail], ex.TRAIL_PEAK_PCT)) if tail.sum() >= 10 else np.nan
        row = {"group": r["group"], "id": r["id"], "sent_idx": r["sent_idx"], "speech_start": r["speech_start"],
               "speech_end": r["speech_end"], "text": r["text"], "noise_floor_db": noise_floor,
               "sentence_peak_db": sent_peak, "tail_peak_db": tail_peak,
               "tail_above_noise_db": tail_peak - noise_floor, "end_drop_db": r["end_drop_db"],
               "terminal_slope_st_s": r["terminal_slope_st_s"]}
        if r["group"] == "strongest final rise":
            row["rise_extent_st"] = r["terminal_slope_st_s"] * ex.TERMINAL_WINDOW_S
        rows.append(row)
    t = pd.DataFrame(rows)
    # Trailing off is audible if the voice drops >= 3 dB (well above the ~1 dB loudness JND) while the tail is
    # still speech (>= 10 dB above background). If the tail is at the noise floor, VAD padded silence.
    tr = t["group"] == "strongest trailing off"
    t.loc[tr, "verdict"] = np.select(
        [t.loc[tr, "tail_above_noise_db"] < 10, t.loc[tr, "end_drop_db"] <= -3],
        ["tail at noise floor: fades out / VAD padding", "audible: voice drops, tail still speech"], "small drop")
    # A final rise is audible if the 300-ms excursion is >= 3 st (intonation rises below ~1.5-3 st are hard to
    # perceive) and the frames are speech-level (the extractor already requires <= 25 dB below the peaks).
    ri = t["group"] == "strongest final rise"
    extent = t.loc[ri, "rise_extent_st"].clip(upper=24)
    t.loc[ri, "verdict"] = np.where(t.loc[ri, "tail_above_noise_db"] < 10, "tail at noise floor",
                                    np.where(extent >= 3, "audible rise (>= 3 st)", "small rise"))
    summary = {g: t[t.group == g]["verdict"].value_counts().to_dict() for g in t.group.unique()}
    return t, summary


# --------------------------------------------------------------------------------------------
# D. Dropout (-300 dB) injection test
# --------------------------------------------------------------------------------------------
def check_dropouts(d: dict) -> tuple[pd.DataFrame, dict]:
    seg = d["seg"]
    info = {"clips_with_any_dropout_in_speech": int((seg["dropout_frac"] > 0).sum()),
            "clips_with_dropout_over_1pct": int((seg["dropout_frac"] > 0.01).sum()),
            "max_dropout_frac": float(seg["dropout_frac"].max())}
    db_cols = {"sentence": ["intensity_mean_db", "intensity_std_db", "end_drop_db"],
               "segment": ["intensity_mean_db", "intensity_std_db", "end_drop_median_db"]}
    ranges = []
    for level, cols in db_cols.items():
        frame = d["sent"] if level == "sentence" else seg
        for c in cols:
            ranges.append({"descriptor": f"{level}.{c}", "min": frame[c].min(), "max": frame[c].max()})
    # Injection: take clean clips, overwrite 5% of their speech intensity frames with -300 dB, re-run the real
    # describe_clip and compare. With the exclusion working, intensity descriptors barely move; a naive mean
    # over the same frames would collapse.
    rng = np.random.default_rng(SEED)
    clean = seg[(seg["dropout_frac"] == 0) & (seg["n_sentences"] >= 3) & ~seg["is_duplicate"]].sample(20, random_state=SEED)
    vad, blips = ex.load_vad(set(clean["id"]))
    rows = []
    for _, r in clean.iterrows():
        clip = d["clips"].loc[r["id"]].copy()
        clip["id"] = r["id"]
        fr = frames_for(r["id"], r["participant"], d["settings"])
        ref = d["settings"].loc[r["participant"], "reference_f0_hz"]
        sett = d["settings"].loc[r["participant"]]
        _, base, _ = ex.describe_clip(clip, vad[r["id"]], fr, ref, sett, blips[r["id"]])
        speech_idx = np.flatnonzero(ex._in_regions(fr["int_t"].astype(float), vad[r["id"]]))
        hit = rng.choice(speech_idx, size=max(1, len(speech_idx) // 20), replace=False)
        corrupted = dict(fr)
        corrupted["int_db"] = fr["int_db"].copy()
        corrupted["int_db"][hit] = -300.0
        _, inj, _ = ex.describe_clip(clip, vad[r["id"]], corrupted, ref, sett, blips[r["id"]])
        naive = float(np.mean(corrupted["int_db"][speech_idx]))
        rows.append({"id": r["id"], "injected_frames": len(hit),
                     "delta_intensity_mean_db": inj["intensity_mean_db"] - base["intensity_mean_db"],
                     "delta_intensity_std_db": inj["intensity_std_db"] - base["intensity_std_db"],
                     "delta_end_drop_median_db": inj["end_drop_median_db"] - base["end_drop_median_db"],
                     "naive_mean_shift_db": naive - base["intensity_mean_db"],
                     "dropout_frac_detected": inj["dropout_frac"]})
    t = pd.DataFrame(rows)
    info["ranges"] = ranges
    info["max_abs_delta_mean_db"] = float(t["delta_intensity_mean_db"].abs().max())
    info["max_abs_delta_std_db"] = float(t["delta_intensity_std_db"].abs().max())
    info["max_abs_delta_end_drop_db"] = float(t["delta_end_drop_median_db"].abs().max())
    info["median_naive_shift_db"] = float(t["naive_mean_shift_db"].median())
    return t, info


# --------------------------------------------------------------------------------------------
# E. Missing values on what Stage 2 will fit on
# --------------------------------------------------------------------------------------------
REASONS = {
    "sent_rate_std_wpm": "needs >= 2 sentences that are >= 4 s and >= 5 words",
    "sent_rate_cv": "needs >= 2 sentences that are >= 4 s and >= 5 words",
    "rushed_opening_ratio": "needs >= 3 sentences, first two and the rest each >= 4 s",
    "rate_drift_ratio": "needs a speech span >= 12 s",
    "monotony_frac": "needs a speech span >= 5 s with >= 1 s voiced per window",
    "sent_rate_mean_wpm": "needs >= 1 sentence >= 4 s and >= 5 words",
    "sent_rate_max_wpm": "needs >= 1 sentence >= 4 s and >= 5 words",
    "terminal_slope_median_st_s": "needs >= 10 strong voiced frames at a sentence end",
}


def check_missing(d: dict) -> pd.DataFrame:
    seg = d["seg"]
    base = seg[(seg["split"] == "train") & ~seg["is_duplicate"]]
    rows = []
    for key, meta in ex.REGISTRY.items():
        if meta["level"] != "segment" or meta["name"] not in base or meta["unit"] == "bool":
            continue
        rows.append({"descriptor": meta["name"], "status": meta["status"],
                     "nan_pct_train_unique": 100 * base[meta["name"]].isna().mean(),
                     "reason": REASONS.get(meta["name"], "no transcript / no VAD speech / too little voicing")})
    return pd.DataFrame(rows).sort_values("nan_pct_train_unique", ascending=False)


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------
def annotate_sheet(d, rates, pitch, aud) -> pd.DataFrame:
    sheet = d["sheet"].copy()
    rate_cols = rates.set_index(["id", "speech_start"])[["whisper_wpm", "whisper_text"]]
    sheet["whisper_wpm"] = [rate_cols["whisper_wpm"].get((i, s), np.nan) for i, s in zip(sheet["id"], sheet["start_s"])]
    sheet["whisper_text"] = [rate_cols["whisper_text"].get((i, s), "") for i, s in zip(sheet["id"], sheet["start_s"])]
    oct_map = pitch.set_index("id")["octave_disagreement_frac"]
    sheet["pyin_octave_disagreement"] = sheet["id"].map(oct_map)
    aud_map = aud.set_index(["id", "speech_start"])["verdict"]
    sheet["auto_verdict"] = [aud_map.get((i, s), "") for i, s in zip(sheet["id"], sheet["start_s"])]
    return sheet


def write_report(rate_sum, pitch_sum, aud_sum, drop_info, missing, rates, pitch, aud) -> Path:
    L = []
    add = L.append
    add("# Stage 1 verification\n")
    add("Automated evidence for the Stage 1 completion checklist (`scripts/explainability/verify_stage1.py`). "
        "Human listening is still required; this narrows down what to listen for. Duplicate recordings excluded.\n")

    add("## Checklist\n")
    add("| # | Item | Automated result | Remaining for a human |")
    add("|---|---|---|---|")
    add(f"| 1 | Listen to the 5 most extreme examples in each of the 7 categories | Listening sheet annotated with Whisper "
        "text/rate, pYIN agreement and audibility verdicts (`listening_check/listening_check_annotated.csv`) | **Listen and fill in `heard_as_extreme`** |")
    add(f"| 2 | Fastest/slowest sentences genuinely representative | Fastest: {rate_sum['fastest_still_extreme']}/{N_EXTREME} still above "
        f"train p95 ({rate_sum['train_p95_wpm']:.0f} WPM) when re-timed by Whisper; slowest: {rate_sum['slowest_still_extreme']}/{N_EXTREME} "
        f"still below p5 ({rate_sum['train_p5_wpm']:.0f} WPM). See §A | Confirm the disagreeing cases by ear |")
    add(f"| 3 | Expressive examples free of pitch-tracking artefacts | Praat vs pYIN octave disagreement in the most expressive answers: "
        f"median {100 * pitch_sum.get('most expressive: median octave disagreement', np.nan):.1f}%, max "
        f"{100 * pitch_sum.get('most expressive: max octave disagreement', np.nan):.1f}% (controls: median "
        f"{100 * pitch_sum.get('control: median octave disagreement', np.nan):.1f}%). See §B | Listen to the top 5 expressive |")
    add(f"| 4 | Final rises and trailing off audible | Trailing off: {aud_sum.get('strongest trailing off', {})}. "
        f"Final rise: {aud_sum.get('strongest final rise', {})}. See §C | Listen to the top 5 of each |")
    add(f"| 5 | -300 dB artefacts cannot enter descriptors | Injection test: max change {drop_info['max_abs_delta_mean_db']:.2f} dB "
        f"(mean), {drop_info['max_abs_delta_std_db']:.2f} dB (std), {drop_info['max_abs_delta_end_drop_db']:.2f} dB (end drop), vs "
        f"{drop_info['median_naive_shift_db']:.1f} dB if they were not excluded. See §D | None |")
    add("| 6 | Review extraction report and missing values | §E | Read `extraction_report.md` |")
    add("")

    add("## A. Speech-rate extremes re-timed with Whisper\n")
    add("Each sentence's VAD-trimmed span is re-counted with faster-whisper word timestamps, independent of the "
        "RecruitView transcript and its 1-s timestamps. `word_ratio` = Whisper words / transcript words in the span.\n")
    add(f"- Control sentences (random {N_CONTROL}): Spearman(transcript WPM, Whisper WPM) = {rate_sum['control_spearman']:.2f}; "
        f"median word ratio {rate_sum['control_median_word_ratio']:.2f}.")
    for g in ("fastest", "slowest"):
        add(f"- {g.capitalize()} {N_EXTREME}: median transcript {rate_sum[f'{g}_median_transcript_wpm']:.0f} WPM vs Whisper "
            f"{rate_sum[f'{g}_median_whisper_wpm']:.0f} WPM; median word ratio {rate_sum[f'{g}_median_word_ratio']:.2f}; "
            f"{rate_sum[f'{g}_still_extreme']} remain extreme.")
    add("")
    add("| group | id | start-end (s) | transcript WPM | Whisper WPM | word ratio | transcript text | Whisper text |")
    add("|---|---|---|---|---|---|---|---|")
    for _, r in rates[rates.group != "control"].head(10).iterrows():
        add(f"| {r.group} | {r.id} | {r.speech_start:.1f}-{r.speech_end:.1f} | {r.transcript_wpm:.0f} | {r.whisper_wpm:.0f} | "
            f"{r.word_ratio:.2f} | {r.transcript_text[:70]} | {r.whisper_text[:70]} |")
    for _, r in rates[rates.group == "slowest"].head(10).iterrows():
        add(f"| {r.group} | {r.id} | {r.speech_start:.1f}-{r.speech_end:.1f} | {r.transcript_wpm:.0f} | {r.whisper_wpm:.0f} | "
            f"{r.word_ratio:.2f} | {r.transcript_text[:70]} | {r.whisper_text[:70]} |")
    add("\nFull table: `verification/rate_extremes_whisper.csv`.\n")

    add("## B. Pitch: Praat vs pYIN\n")
    add("`octave_disagreement_frac` = share of frames voiced in both trackers whose F0 differs by > 6 st "
        "(an octave error in one of them).\n")
    add("| Measure | Value |")
    add("|---|---|")
    for k, v in pitch_sum.items():
        add(f"| {k} | {v:.3f} |")
    add("")
    add("| group | id | speaker | Praat std (st) | pYIN std (st) | octave disagreement | median abs diff (st) |")
    add("|---|---|---|---|---|---|---|")
    for _, r in pitch[pitch.group != "control"].iterrows():
        add(f"| {r.group} | {r.id} | {r.participant} | {r.praat_pitch_std_st:.2f} | {r.pyin_pitch_std_st:.2f} | "
            f"{100 * r.octave_disagreement_frac:.1f}% | {r.median_abs_diff_st:.2f} |")
    add("")

    add("## C. Audibility of trailing off and final rises\n")
    add("`tail_above_noise_db` = syllable-peak level in the last 0.7 s minus the clip's background level (p10 of "
        "non-speech frames). < 10 dB means the 'speech' at the end is really background. Trailing off counts as "
        "audible at a drop of >= 3 dB with the tail still speech; a final rise at >= 3 st over 300 ms.\n")
    add("| group | id | span (s) | end drop (dB) | rise (st/s) | tail above noise (dB) | verdict | text |")
    add("|---|---|---|---|---|---|---|---|")
    for _, r in aud.groupby("group").head(8).iterrows():
        add(f"| {r.group} | {r.id} | {r.speech_start:.1f}-{r.speech_end:.1f} | {r.end_drop_db:.1f} | {r.terminal_slope_st_s:.1f} | "
            f"{r.tail_above_noise_db:.1f} | {r.verdict} | {str(r.text)[:60]} |")
    add("\nFull table: `verification/audibility.csv`.\n")

    add("## D. Digital-silence dropouts (-300 dB)\n")
    add(f"- Clips with any dropout inside VAD speech: {drop_info['clips_with_any_dropout_in_speech']}; with > 1% of speech "
        f"frames: {drop_info['clips_with_dropout_over_1pct']}; max {100 * drop_info['max_dropout_frac']:.1f}%.")
    add("- Code path: every intensity statistic uses `int_ok = speech & ~dropout`; the intensity sampled at pitch "
        "frames (terminal-slope guard) sets dropout frames to NaN; QC plots now mask them and mark them with a rug.")
    add(f"- Injection test (20 clean clips, 5% of speech frames set to -300 dB, real `describe_clip` re-run): max change "
        f"{drop_info['max_abs_delta_mean_db']:.2f} dB in mean intensity, {drop_info['max_abs_delta_std_db']:.2f} dB in intensity std, "
        f"{drop_info['max_abs_delta_end_drop_db']:.2f} dB in end drop. A naive mean over the same frames would shift by "
        f"{drop_info['median_naive_shift_db']:.1f} dB.")
    add("- Observed ranges of every dB descriptor:\n")
    add("| descriptor | min | max |")
    add("|---|---|---|")
    for r in drop_info["ranges"]:
        add(f"| `{r['descriptor']}` | {r['min']:.1f} | {r['max']:.1f} |")
    add("")

    add("## E. Missing values (train, unique recordings, answer level)\n")
    add("| descriptor | status | NaN % | reason when missing |")
    add("|---|---|---|---|")
    for _, r in missing.iterrows():
        add(f"| `{r.descriptor}` | {r.status} | {r.nan_pct_train_unique:.1f} | {r.reason} |")
    add("")
    path = P.VERIFICATION_REPORT
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--whisper-model", default="small.en")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    VER_DIR.mkdir(parents=True, exist_ok=True)
    d = load()
    print("A. rate extremes vs Whisper...")
    rates, rate_sum = check_rates(d, args.whisper_model)
    rates.to_csv(VER_DIR / "rate_extremes_whisper.csv", index=False)
    print("B. pitch vs pYIN...")
    pitch, pitch_sum = check_pitch(d, args.workers)
    pitch.to_csv(VER_DIR / "pitch_crosscheck.csv", index=False)
    print("C. audibility...")
    aud, aud_sum = check_audibility(d)
    aud.to_csv(VER_DIR / "audibility.csv", index=False)
    print("D. dropouts...")
    drops, drop_info = check_dropouts(d)
    drops.to_csv(VER_DIR / "dropout_injection.csv", index=False)
    print("E. missing values...")
    missing = check_missing(d)
    missing.to_csv(VER_DIR / "missing_values.csv", index=False)
    annotate_sheet(d, rates, pitch, aud).to_csv(P.LISTENING / "listening_check_annotated.csv", index=False)
    report = write_report(rate_sum, pitch_sum, aud_sum, drop_info, missing, rates, pitch, aud)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rate_sum.items()}, indent=1))
    print(json.dumps({k: round(v, 3) for k, v in pitch_sum.items()}, indent=1))
    print(aud_sum)
    print({k: v for k, v in drop_info.items() if k != "ranges"})
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
