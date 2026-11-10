"""Stage 1: measure delivery descriptors per sentence and per answer (see markdown/EXPLAINABILITY_STAGES.md).

MEASUREMENT ONLY. This script produces raw, physically interpretable values ("the speaker spoke at 195 WPM
in this sentence"). It deliberately does **not** compare them with anyone (no percentiles, no speaker
baselines, no question-type ratios) and does not judge them (no "too fast"). Those are separate layers:

    measurement  (Stage 1, this script)   "spoke at 195 WPM"
    comparison   (Stage 2)                "above the 90th percentile of comparable training answers,
                                           +28% vs the speaker's own other answers"
    feedback     (Stage 9, LLM)           "your pace was relatively fast; slowing slightly during complex
                                           explanations may improve clarity"

Keeping them apart means the reference distributions (fitted on TRAIN only) can be changed without
re-measuring, and every feedback sentence can be traced back to a number.

Every descriptor has a status in results/delivery_descriptors/descriptor_registry.json:
  active        may drive flags and feedback once Stage 2/3 have normed and validated it
  experimental  measured and normed, analysed in Stage 3, but NOT used for feedback yet (pitch-based
                measures, trailing off, voice quality: Stage 0 found no rater support or no validation yet)
  diagnostic    quality control / covariates only, never feedback (absolute loudness and absolute pitch:
                Stage 0 showed they mostly identify the speaker or the microphone)

Inputs (nothing is re-run except Praat):
  Datasets/RecruitView Audio/{split}_audio_metadata.csv   audio paths + transcripts with [MM:SS - MM:SS]
  results/vad_pause/features/{split}_vad_regions.csv      cached Silero VAD regions
  Datasets/question_types.csv                             Stage 0

Timing model. RecruitView transcript timestamps have 1-s resolution and are floored: against VAD, speech
starts a median 0.7 s after the stamped start and ends 0.9 s after the stamped end. A line [a - b] is
therefore treated as the window [a, b+1) (capped at the next line's start), and then trimmed to the VAD
speech inside it, which gives ~0.1 s precision for sentence duration. Mid-sentence vs between-sentence
pauses are still limited by the 1-s stamps (a +-0.5 s tolerance zone around each boundary); Stage 4 word
alignment replaces this.

Pitch. Two Praat passes: pass 1 at 75-600 Hz pools all of a speaker's clips; pass 2 uses
floor = 0.75 x q15 and ceiling = 1.5 x q85 of that speaker's F0 (avoids octave errors without guessing sex).
F0 is expressed in semitones relative to the speaker's median F0. Participants are split-disjoint, and the
same procedure applies to a new user's own interview, so this uses no labels and leaks nothing.

Outputs (results/delivery_descriptors/):
  sentence_descriptors.csv   one row per transcript sentence (all splits)
  segment_descriptors.csv    one row per answer clip (all splits)
  pause_events.csv           every VAD gap with its class (leading / mid_sentence / between_sentence / ...)
  speaker_pitch_settings.csv per-speaker pitch floor, ceiling and reference F0
  descriptor_registry.json   category, status, unit and definition of every descriptor
  listening_check.csv        extreme clips/sentences for the manual listening check
  qc_plots/*.png             pitch / intensity / VAD / pause-class plots for a few extreme clips
  extraction_report.md       coverage, NaN rates, sanity checks

Run:  python scripts/explainability/extract_delivery_descriptors.py [--workers 8] [--limit 20]
"""
from __future__ import annotations

import argparse
import sys
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AUDIO_META_DIR = ROOT / "Datasets" / "RecruitView Audio"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402  (results layout)
VAD_DIR = P.VAD_FEATURES
QUESTION_TYPES = ROOT / "Datasets" / "question_types.csv"
OUT_DIR = P.S1
CACHE_DIR = ROOT / "outputs" / "audio" / "cache" / "delivery_descriptors"
SPLITS = ["train", "val", "test"]

CACHE_VERSION = "v1"
# Measurement definitions (fixed physical/definitional constants, not fitted thresholds).
MIN_RATE_SENTENCE_S = 4.0      # per-sentence rates only for sentences >= 4 s (1-s stamps make shorter ones noisy)
TIMESTAMP_PAD_S = 1.0          # stamps are floored: [a - b] covers [a, b+1)
BOUNDARY_TOL_S = 0.5           # tolerance around a sentence boundary for between-sentence pauses
MIN_PAUSE_S = 0.25             # perceptual pause threshold for counting pauses
LONG_PAUSE_S = 1.0             # "long" mid-sentence pause
PITCH_PASS1 = (75.0, 600.0)
PITCH_STEP_S = 0.01
PITCH_FLOOR_X_MEDIAN = 0.6     # speaker pitch floor never below 0.6 x speaker median F0 (excludes halving errors)
PITCH_CEILING_X_MEDIAN = 1.8   # ceiling never above 1.8 x median (excludes doubling errors)
INTENSITY_MIN_PITCH = 75.0     # fixed for every speaker so intensity frames are comparable
MIN_VOICED_FRAMES = 30         # >= 0.3 s of voiced frames for a pitch statistic
MIN_INTENSITY_FRAMES = 30
TRAIL_WINDOW_S = 0.7           # trailing off: last 0.7 s of a sentence's speech vs the sentence mean
MIN_TRAIL_SENTENCE_SPEECH_S = 1.5
TRAIL_PEAK_PCT = 80            # trailing off compares syllable-peak level (80th percentile of intensity)
TRAIL_MIN_ABOVE_NOISE_DB = 10.0  # tail peaks within 10 dB of the clip background are silence (VAD padding), not speech
TERMINAL_WINDOW_S = 0.3        # terminal F0 slope: last 300 ms of voicing
MIN_RATE_WORDS = 5             # ... and >= 5 words (rates on 2-word lines like "Thank you." are meaningless)
MAX_PLAUSIBLE_SPS = 8.0        # articulation above 8 syllables/s is not humanly sustained -> timestamps/text mismatch
DROPOUT_DB = -100.0            # intensity frames at/below this are digital silence (exact zeros), not speech
BLIP_MAX_S = 0.3               # VAD speech islands shorter than this ...
BLIP_ISOLATION_S = 1.0         # ... with >= 1 s of silence on both sides are clicks/breaths, removed
MIN_TERMINAL_FRAMES = 10
MIN_VOICED_RUN = 5             # terminal window ends at the last run of >= 5 consecutive voiced frames (50 ms)
TERMINAL_MAX_BELOW_PEAK_DB = 25.0  # terminal-slope frames must be within 25 dB of the sentence syllable-peak level
MONO_WINDOW_S, MONO_HOP_S = 5.0, 2.5
MONO_MIN_VOICED_FRAMES = 100   # >= 1 s voiced in a 5-s window
MONO_STD_ST = 1.5              # a window with pitch std < 1.5 st counts as monotone (definition from the README)
OCTAVE_JUMP_ST = 9.0           # adjacent voiced frames differing by > 9 st = likely tracking error

TIMESTAMP_LINE = re.compile(r"\[(\d+):(\d+)\s*-\s*(\d+):(\d+)\]\s*(.*)")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


# ============================================================================================
# Descriptor registry: the single place that says what each column means and whether it may be
# used for feedback. Stage 2 reads this.
# ============================================================================================
A, E, D = "active", "experimental", "diagnostic"
REGISTRY: dict[str, dict] = {}


def _reg(level: str, name: str, category: str, status: str, unit: str, definition: str) -> None:
    REGISTRY[f"{level}.{name}"] = {"level": level, "name": name, "category": category, "status": status,
                                   "unit": unit, "definition": definition}


for lvl in ("sentence", "segment"):
    pre = "sentence" if lvl == "sentence" else "answer"
    _reg(lvl, "speech_rate_wpm", "pace", A, "words/min",
         f"Words / {pre} duration (VAD-trimmed, includes internal pauses). "
         + (f"Only for sentences >= {MIN_RATE_SENTENCE_S:.0f} s and >= {MIN_RATE_WORDS} words." if lvl == "sentence"
            else "Duration = first to last speech."))
    _reg(lvl, "artic_rate_sps", "pace", A, "syllables/s",
         f"Articulation rate: syllables / VAD speech time within the {pre} (pauses excluded). Syllables from an "
         "orthographic heuristic on the cleaned transcript.")
    _reg(lvl, "artic_rate_wpm", "pace", A, "words/min", f"Words / VAD speech time within the {pre}.")
    _reg(lvl, "pitch_std_st", "expressiveness", E, "semitones", f"Std of voiced F0 (st re. speaker median) in the {pre}.")
    _reg(lvl, "pitch_range_st", "expressiveness", E, "semitones", f"p95 - p5 of voiced F0 (st) in the {pre}.")
    _reg(lvl, "f0_median_st", "composure", E, "semitones",
         f"Median voiced F0 of the {pre} relative to the speaker's overall median (raised pitch = stress proxy).")
    _reg(lvl, "intensity_std_db", "expressiveness", E, "dB",
         f"Std of Praat intensity over VAD-speech frames in the {pre}. Gain-invariant (energy variability, not level).")
    _reg(lvl, "intensity_mean_db", "diagnostic", D, "dB",
         f"Mean intensity over speech frames. ABSOLUTE LEVEL: depends on microphone/gain; never feedback (Stage 0).")
    _reg(lvl, "voiced_frac", "diagnostic", D, "fraction", f"Voiced pitch frames / speech frames in the {pre}.")

_reg("sentence", "duration_s", "conciseness", A, "s", "VAD-trimmed sentence duration (first to last speech in the window).")
_reg("sentence", "speech_time_s", "conciseness", A, "s", "VAD speech time inside the sentence window.")
_reg("sentence", "n_words", "conciseness", A, "count", "Words in the sentence (cleaned transcript; fillers removed by RecruitView).")
_reg("sentence", "n_mid_pauses", "pausing", A, "count", f"Mid-sentence VAD pauses >= {MIN_PAUSE_S} s in this sentence.")
_reg("sentence", "max_mid_pause_s", "pausing", A, "s", "Longest mid-sentence pause in this sentence (0 if none).")
_reg("sentence", "mid_pause_time_s", "pausing", A, "s", f"Total mid-sentence pause time (pauses >= {MIN_PAUSE_S} s).")
_reg("sentence", "end_drop_db", "projection", E, "dB",
     f"Syllable-peak level (p{TRAIL_PEAK_PCT} intensity) in the last {TRAIL_WINDOW_S} s of the sentence's speech minus "
     "the sentence's p{TRAIL_PEAK_PCT} (negative = trailing off). Gain-invariant; dropout frames excluded; NaN when the tail is within 10 dB of the clip background (VAD padding, not speech).".replace("{TRAIL_PEAK_PCT}", str(TRAIL_PEAK_PCT)))
_reg("sentence", "terminal_slope_st_s", "assertiveness", E, "st/s",
     f"Linear F0 slope over the {TERMINAL_WINDOW_S} s before the end of the sentence's last voiced run "
     f"(>= {MIN_VOICED_RUN} consecutive voiced frames; positive = rising / uptalk).")

_reg("segment", "audio_duration_s", "conciseness", A, "s", "Length of the audio file.")
_reg("segment", "speech_span_s", "conciseness", A, "s", "First to last VAD speech (answer length without leading/trailing silence).")
_reg("segment", "speech_time_s", "conciseness", A, "s", "Total VAD speech time.")
_reg("segment", "word_count", "conciseness", A, "count", "Words in the cleaned transcript.")
_reg("segment", "syllable_count", "conciseness", D, "count", "Heuristic syllables (used for articulation rate).")
_reg("segment", "n_sentences", "conciseness", A, "count", "Timestamped transcript sentences.")
_reg("segment", "sent_rate_mean_wpm", "pace", A, "words/min", "Mean per-sentence speech rate over sentences >= 4 s.")
_reg("segment", "sent_rate_max_wpm", "pace", A, "words/min", "Fastest sentence (>= 4 s).")
_reg("segment", "sent_rate_std_wpm", "pace", A, "words/min", "Rate variability: std of per-sentence rates (needs >= 2 sentences >= 4 s).")
_reg("segment", "sent_rate_cv", "pace", A, "ratio", "Rate variability normalised by the mean (std / mean).")
_reg("segment", "n_rate_sentences", "pace", D, "count", "Sentences >= 4 s (eligible for per-sentence rate).")
_reg("segment", "rushed_opening_ratio", "pace", A, "ratio",
     "Speech rate of the first 2 sentences / rate of the rest (needs >= 3 sentences, each part >= 4 s). > 1 = faster opening.")
_reg("segment", "rate_drift_ratio", "pace", A, "ratio",
     "Speech rate in the last third of the answer / first third (by time; needs a speech span >= 12 s). < 1 = slowing down.")
_reg("segment", "response_latency_s", "pausing", A, "s", "Silence before the first VAD speech.")
_reg("segment", "pause_ratio", "pausing", A, "fraction", "Internal non-speech time / speech span (all VAD gaps).")
_reg("segment", "mid_pause_rate_per_min", "pausing", A, "per min", f"Mid-sentence pauses >= {MIN_PAUSE_S} s per minute of speech span.")
_reg("segment", "long_mid_pause_rate_per_min", "pausing", A, "per min", f"Mid-sentence pauses >= {LONG_PAUSE_S} s per minute.")
_reg("segment", "mid_pause_max_s", "pausing", A, "s", "Longest mid-sentence pause in the answer (0 if none).")
_reg("segment", "mid_pause_time_ratio", "pausing", A, "fraction", "Mid-sentence pause time / speech span.")
_reg("segment", "between_pause_time_ratio", "pausing", A, "fraction", "Between-sentence pause time / speech span.")
_reg("segment", "internal_pause_count", "pausing", D, "count",
     f"Raw count of internal pauses >= {MIN_PAUSE_S} s. Tracks answer length (Stage 0); use the rates instead.")
_reg("segment", "monotony_frac", "expressiveness", E, "fraction",
     f"Fraction of {MONO_WINDOW_S:.0f}-s windows (hop {MONO_HOP_S} s, >= 1 s voiced) with pitch std < {MONO_STD_ST} st.")
_reg("segment", "n_monotony_windows", "expressiveness", D, "count", "Windows that entered monotony_frac.")
_reg("segment", "end_drop_median_db", "projection", E, "dB", "Median sentence end_drop_db over the answer.")
_reg("segment", "terminal_slope_median_st_s", "assertiveness", E, "st/s", "Median sentence terminal F0 slope.")
_reg("segment", "jitter_local", "composure", E, "fraction", "Praat jitter (local) over the clip.")
_reg("segment", "shimmer_local", "composure", E, "fraction", "Praat shimmer (local) over the clip.")
_reg("segment", "f0_median_hz", "diagnostic", D, "Hz", "Median F0 in Hz. ABSOLUTE: identifies the voice; never feedback.")
_reg("segment", "octave_jump_frac", "diagnostic", D, "fraction",
     f"Adjacent voiced frames differing by > {OCTAVE_JUMP_ST:.0f} st (pitch-tracking error rate).")
_reg("segment", "dropout_frac", "diagnostic", D, "fraction",
     "Fraction of VAD-speech intensity frames that are digital silence (recording dropouts). Input to the Stage 6 quality gate.")
_reg("segment", "isolated_blips_removed", "diagnostic", D, "count",
     f"VAD speech islands < {BLIP_MAX_S} s isolated by >= {BLIP_ISOLATION_S} s of silence, removed before measuring.")
_reg("segment", "transcript_timing_suspect", "diagnostic", D, "bool",
     f"Some transcript line implies > {MAX_PLAUSIBLE_SPS:.0f} syllables/s (text does not match its timestamps); "
     "affected rates are withheld.")
_reg("segment", "is_duplicate", "diagnostic", D, "bool",
     "Byte-identical audio to a lower id (RecruitView duplicate recording). Exclude when fitting reference distributions.")
_reg("sentence", "timing_suspect", "diagnostic", D, "bool",
     f"Line implies > {MAX_PLAUSIBLE_SPS:.0f} syllables/s; its rates are withheld.")
_reg("segment", "pitch_floor_hz", "diagnostic", D, "Hz", "Speaker-specific Praat pitch floor used.")
_reg("segment", "pitch_ceiling_hz", "diagnostic", D, "Hz", "Speaker-specific Praat pitch ceiling used.")


# ============================================================================================
# Text
# ============================================================================================
def count_syllables(token: str) -> int:
    """Orthographic syllable estimate. Approximate (~85-90% word accuracy for English); Stage 4 can swap in
    CMUdict on aligned words. Acronyms count one syllable per letter, digits one per digit."""
    if token.isdigit():
        return len(token)
    if len(token) >= 2 and token.isupper():
        return len(token)
    word = re.sub(r"[^a-z]", "", token.lower())
    if not word:
        return 0
    n = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    elif word.endswith("ed") and not word.endswith(("ted", "ded")) and n > 1:
        n -= 1
    return max(1, n)


def parse_sentences(transcript: str, audio_duration: float) -> list[dict]:
    lines = []
    for line in (transcript or "").splitlines():
        m = TIMESTAMP_LINE.match(line.strip())
        if m:
            start, end = int(m[1]) * 60 + int(m[2]), int(m[3]) * 60 + int(m[4])
            lines.append((float(start), float(end), m[5].strip()))
    sentences = []
    for i, (start, end, text) in enumerate(lines):
        win_end = end + TIMESTAMP_PAD_S
        if i + 1 < len(lines):
            win_end = max(end, min(win_end, lines[i + 1][0]))
        win_end = min(win_end, audio_duration)
        win_start = min(start, audio_duration)
        tokens = WORD_RE.findall(text)
        sentences.append({"sent_idx": i, "ts_start": start, "ts_end": end, "win_start": win_start,
                          "win_end": max(win_start, win_end), "text": text, "n_words": len(tokens),
                          "n_syllables": sum(count_syllables(t) for t in tokens)})
    return sentences


# ============================================================================================
# Praat (worker processes)
# ============================================================================================
def _pass1(job: tuple[str, str]) -> tuple[str, np.ndarray]:
    """Pitch at 75-600 Hz; returns the full F0 track (0 = unvoiced). Cached."""
    clip_id, audio_path = job
    cache = CACHE_DIR / "pass1" / f"{clip_id}_{CACHE_VERSION}.npy"
    if cache.exists():
        return clip_id, np.load(cache)
    import parselmouth
    snd = parselmouth.Sound(audio_path)
    f0 = snd.to_pitch(time_step=PITCH_STEP_S, pitch_floor=PITCH_PASS1[0],
                      pitch_ceiling=PITCH_PASS1[1]).selected_array["frequency"].astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, f0)
    return clip_id, f0


def _pass2(job: tuple[str, str, float, float]) -> tuple[str, dict]:
    """Speaker-specific pitch, intensity, jitter, shimmer. Cached per (floor, ceiling)."""
    clip_id, audio_path, floor, ceiling = job
    cache = CACHE_DIR / "pass2" / f"{clip_id}_{floor:.0f}_{ceiling:.0f}_{CACHE_VERSION}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return clip_id, {k: z[k] for k in z.files}
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch(time_step=PITCH_STEP_S, pitch_floor=floor, pitch_ceiling=ceiling)
    intensity = snd.to_intensity(minimum_pitch=INTENSITY_MIN_PITCH, time_step=PITCH_STEP_S)
    jitter = shimmer = np.nan
    try:
        points = call(snd, "To PointProcess (periodic, cc)", floor, ceiling)
        jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:  # very short / unvoiced clips
        pass
    out = {"pitch_t": pitch.xs().astype(np.float32),
           "f0": pitch.selected_array["frequency"].astype(np.float32),
           "int_t": intensity.xs().astype(np.float32),
           "int_db": intensity.values[0].astype(np.float32),
           "jitter": np.float32(jitter), "shimmer": np.float32(shimmer)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return clip_id, out


def run_parallel(func, jobs: list, workers: int, label: str) -> dict:
    results, start = {}, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(func, job) for job in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            clip_id, value = fut.result()
            results[clip_id] = value
            if k % 200 == 0 or k == len(jobs):
                print(f"  {label}: {k}/{len(jobs)} ({time.time() - start:.0f}s)", flush=True)
    return results


# ============================================================================================
# Loading
# ============================================================================================
def load_clips(limit: int | None) -> pd.DataFrame:
    frames = []
    for split in SPLITS:
        meta = pd.read_csv(AUDIO_META_DIR / f"{split}_audio_metadata.csv", dtype={"id": str})
        frames.append(meta.assign(split=split))
    meta = pd.concat(frames, ignore_index=True)
    meta = meta[meta["status"] == "extracted"]
    qtypes = pd.read_csv(QUESTION_TYPES)[["question_id", "question_type"]]
    meta = meta.merge(qtypes, on="question_id", how="left")
    clips = pd.DataFrame({
        "id": meta["id"], "split": meta["split"], "participant": meta["user_id"].astype(str),
        "question_id": meta["question_id"], "question_type": meta["question_type"],
        "duration_bucket": meta["duration"], "video_quality": meta["video_quality"],
        "audio_path": meta["audio_path"].map(lambda p: str(ROOT / Path(str(p).replace("\\", "/")))),
        "audio_duration_s": meta["duration_seconds"].astype(float),
        "transcript": meta["transcript"].fillna(""),
    })
    if limit:
        keep = clips["participant"].drop_duplicates().head(max(1, limit // 5))
        clips = clips[clips["participant"].isin(keep)].head(limit)
    return clips.reset_index(drop=True)


def remove_isolated_blips(regions: np.ndarray) -> tuple[np.ndarray, int]:
    """Drop VAD speech islands < BLIP_MAX_S with >= BLIP_ISOLATION_S of silence on both sides (file edges count
    as silence). These are clicks, breaths or noise; left in, they stretch sentence spans and turn trailing
    silence into fake mid-sentence pauses."""
    if len(regions) == 0:
        return regions, 0
    before = np.r_[np.inf, regions[1:, 0] - regions[:-1, 1]]
    after = np.r_[regions[1:, 0] - regions[:-1, 1], np.inf]
    blip = ((regions[:, 1] - regions[:, 0]) < BLIP_MAX_S) & (before >= BLIP_ISOLATION_S) & (after >= BLIP_ISOLATION_S)
    if blip.all():  # never delete all speech
        return regions, 0
    return regions[~blip], int(blip.sum())


def load_vad(ids: set[str]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    regions = pd.concat([pd.read_csv(VAD_DIR / f"{s}_vad_regions.csv", dtype={"id": str}) for s in SPLITS])
    speech = regions[(regions["region_type"] == "speech") & regions["id"].isin(ids)]
    out = {i: np.empty((0, 2)) for i in ids}
    blips = {i: 0 for i in ids}
    for clip_id, g in speech.groupby("id"):
        out[clip_id], blips[clip_id] = remove_isolated_blips(
            g.sort_values("start_seconds")[["start_seconds", "end_seconds"]].to_numpy(float))
    return out, blips


def duplicate_groups(clips: pd.DataFrame) -> pd.DataFrame:
    """RecruitView contains byte-identical recordings under different ids (same participant + question),
    rated independently. Mark them so Stage 2 can fit reference distributions on unique recordings."""
    import hashlib
    digests = clips["audio_path"].map(lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest())
    out = pd.DataFrame({"id": clips["id"], "audio_md5": digests})
    out["duplicate_group_size"] = out.groupby("audio_md5")["id"].transform("size")
    out["duplicate_of"] = out.groupby("audio_md5")["id"].transform("min")
    out["is_duplicate"] = out["id"] != out["duplicate_of"]
    return out


# ============================================================================================
# Speaker pitch settings
# ============================================================================================
def speaker_settings(clips: pd.DataFrame, pass1: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for participant, g in clips.groupby("participant"):
        voiced = np.concatenate([pass1[i][pass1[i] > 0] for i in g["id"]])
        # De Looze & Hirst (2008) quartile rule: floor = 0.75 x Q1, ceiling = 1.5 x Q3. The stage guide's
        # q15/q85 variant was tried first, but pass-1 octave errors contaminate q85 (speaker 315: q85 = 3.5 x
        # median -> 569 Hz ceiling), while the quartiles are robust to them.
        # The quartiles themselves can be contaminated when a speaker has many halving errors (speaker 97:
        # Q1 = 121 Hz vs median 200 Hz -> floor 91 Hz lets Praat track a whole octave down). So the range is
        # also anchored to the median: floor >= 0.6 x median (-8.8 st), ceiling <= 1.8 x median (+10.2 st).
        # Halving (-12 st) and doubling (+12 st) errors then fall outside the search range.
        if len(voiced) >= 200:
            q25, q50, q75 = np.percentile(voiced, [25, 50, 75])
            floor = float(np.clip(round(max(0.75 * q25, PITCH_FLOOR_X_MEDIAN * q50)), 50, 250))
            ceiling = float(np.clip(round(min(1.5 * q75, PITCH_CEILING_X_MEDIAN * q50)), floor + 100, 700))
            source = "speaker"
        else:
            q25 = q50 = q75 = np.nan
            floor, ceiling, source = PITCH_PASS1[0], PITCH_PASS1[1], "fallback"
        rows.append({"participant": participant, "n_clips": len(g), "pass1_voiced_frames": len(voiced),
                     "pass1_q25_hz": q25, "pass1_median_hz": q50, "pass1_q75_hz": q75, "pitch_floor_hz": floor,
                     "pitch_ceiling_hz": ceiling, "source": source})
    return pd.DataFrame(rows)


# ============================================================================================
# Per-clip descriptors
# ============================================================================================
def _in_regions(t: np.ndarray, regions: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(t), dtype=bool)
    for s, e in regions:
        mask |= (t >= s) & (t <= e)
    return mask


def _clip_regions(regions: np.ndarray, start: float, end: float) -> np.ndarray:
    if len(regions) == 0 or end <= start:
        return np.empty((0, 2))
    clipped = np.column_stack([np.maximum(regions[:, 0], start), np.minimum(regions[:, 1], end)])
    return clipped[clipped[:, 1] > clipped[:, 0]]


def _octave_jump_frac(f0: np.ndarray) -> float:
    voiced = f0 > 0
    both = voiced[1:] & voiced[:-1]
    if both.sum() < 20:
        return np.nan
    st = 12 * np.log2(np.where(voiced, f0, 1.0))
    return float((np.abs(np.diff(st))[both] > OCTAVE_JUMP_ST).mean())


def _last_voiced_run_end(mask: np.ndarray, t: np.ndarray, min_run: int = MIN_VOICED_RUN) -> float:
    """End time of the last run of >= min_run consecutive voiced frames (ignores isolated voicing blips)."""
    idx = np.flatnonzero(mask)
    if len(idx) < min_run:
        return np.nan
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    long_runs = np.flatnonzero(ends - starts + 1 >= min_run)
    return float(t[ends[long_runs[-1]]]) if len(long_runs) else np.nan


def _pitch_stats(st: np.ndarray) -> tuple[float, float, float]:
    if len(st) < MIN_VOICED_FRAMES:
        return np.nan, np.nan, np.nan
    p5, p95 = np.percentile(st, [5, 95])
    return float(np.std(st)), float(p95 - p5), float(np.median(st))


def classify_pauses(gaps: np.ndarray, sentences: list[dict], spans: list[tuple[float, float]]) -> list[tuple[str, int]]:
    """Class + owning sentence for each internal VAD gap.

    mid_sentence      inside one sentence's VAD-trimmed speech span and away from every sentence boundary
    between_sentence  overlaps a boundary zone [end_i - tol, start_{i+1} + 1 + tol], or lies between two
                      sentences' speech spans (e.g. silence inside a stretched timestamp window)
    outside_transcript before the first / after the last transcribed sentence (untranscribed speech around it)
    unclassified      clip has no timestamped transcript
    """
    zones = [(sentences[i]["ts_end"] - BOUNDARY_TOL_S, sentences[i + 1]["ts_start"] + TIMESTAMP_PAD_S + BOUNDARY_TOL_S)
             for i in range(len(sentences) - 1)]
    first_start = min((a for a, _ in spans), default=np.nan)
    last_end = max((b for _, b in spans), default=np.nan)
    out = []
    for g0, g1 in gaps:
        if not sentences:
            out.append(("unclassified", -1))
            continue
        if any(g0 < z1 and g1 > z0 for z0, z1 in zones):
            out.append(("between_sentence", -1))
            continue
        owner = next((s["sent_idx"] for s, (a, b) in zip(sentences, spans) if a <= g0 and g1 <= b), -1)
        if owner >= 0:
            out.append(("mid_sentence", owner))
        elif g1 <= first_start or g0 >= last_end:
            out.append(("outside_transcript", -1))
        else:
            out.append(("between_sentence", -1))
    return out


def _octave_fold(st: np.ndarray, centre: float) -> np.ndarray:
    """Fold frames that sit > OCTAVE_JUMP_ST from `centre` back by whole octaves (halving/doubling errors,
    common in creaky voice at sentence ends)."""
    off = st - centre
    shift = np.where(np.abs(off) > OCTAVE_JUMP_ST, 12.0 * np.round(off / 12.0), 0.0)
    return st - shift


def _terminal_slope(t: np.ndarray, st: np.ndarray, sentence_median_st: float) -> float:
    """Robust terminal F0 slope: octave-fold against the sentence median, drop frames still > 6 st from the
    window median, then Theil-Sen."""
    st = _octave_fold(st, sentence_median_st)
    keep = np.abs(st - np.median(st)) <= 6.0
    if keep.sum() < MIN_TERMINAL_FRAMES:
        return np.nan
    from scipy.stats import theilslopes
    return float(theilslopes(st[keep], t[keep])[0])


def classify_pauses_by_words(gaps: np.ndarray, sentences: list[dict]) -> list[tuple[str, int]]:
    """Stage 4 rule, for sentences built from ASR words (each sentence carries `word_mids`, the midpoints of its
    words). Word midpoints are used instead of word edges because Whisper stretches word ends into silence.

    mid_sentence       the last word before the gap and the first word after it belong to the same sentence
    between_sentence   they belong to different sentences
    outside_transcript no recognised word before or after the gap
    """
    mids = np.concatenate([np.asarray(s["word_mids"], float) for s in sentences]) if sentences else np.empty(0)
    owner = np.concatenate([np.full(len(s["word_mids"]), s["sent_idx"]) for s in sentences]) if sentences else np.empty(0)
    order = np.argsort(mids, kind="stable")
    mids, owner = mids[order], owner[order]
    out = []
    for g0, g1 in gaps:
        if not len(mids):
            out.append(("unclassified", -1))
            continue
        centre = (g0 + g1) / 2
        k = int(np.searchsorted(mids, centre))
        if k == 0 or k == len(mids):
            out.append(("outside_transcript", -1))
        elif owner[k - 1] == owner[k]:
            out.append(("mid_sentence", int(owner[k])))
        else:
            out.append(("between_sentence", -1))
    return out


def describe_clip(clip: pd.Series, regions: np.ndarray, frames: dict, ref_hz: float,
                  settings: pd.Series, n_blips: int, sentences: list[dict] | None = None
                  ) -> tuple[list[dict], dict, list[dict]]:
    """`sentences=None` (Stage 1): sentences = timestamped transcript lines. Stage 4 passes sentences built from
    verbatim ASR words (same keys plus `word_mids`); pauses are then classified by word boundaries."""
    word_mode = sentences is not None
    if sentences is None:
        sentences = parse_sentences(clip["transcript"], clip["audio_duration_s"])
    pt, f0 = frames["pitch_t"].astype(float), frames["f0"].astype(float)
    it, idb = frames["int_t"].astype(float), frames["int_db"].astype(float)
    voiced = f0 > 0
    st_all = np.full_like(f0, np.nan)
    st_all[voiced] = 12 * np.log2(f0[voiced] / ref_hz)
    int_speech = _in_regions(it, regions)
    dropout = idb <= DROPOUT_DB          # digital silence (exact zeros) -> Praat reports -300 dB
    int_ok = int_speech & ~dropout       # intensity frames used for every intensity statistic
    background = idb[~int_speech & ~dropout]
    noise_floor = float(np.percentile(background, 10)) if len(background) >= 20 else -np.inf
    pitch_speech = _in_regions(pt, regions)
    int_at_pitch = np.interp(pt, it, np.where(dropout, np.nan, idb))  # intensity at each pitch frame

    has_speech = len(regions) > 0
    span_start = regions[0, 0] if has_speech else np.nan
    span_end = regions[-1, 1] if has_speech else np.nan
    span = span_end - span_start if has_speech else np.nan

    # ---- sentence speech spans (timestamp window trimmed to VAD speech)
    spans, pieces_of = [], []
    for s in sentences:
        pieces = _clip_regions(regions, s["win_start"], s["win_end"])
        pieces_of.append(pieces)
        spans.append((float(pieces[0, 0]), float(pieces[-1, 1])) if len(pieces) else (s["win_start"], s["win_end"]))

    # ---- pauses
    gaps = np.column_stack([regions[:-1, 1], regions[1:, 0]]) if len(regions) > 1 else np.empty((0, 2))
    classes = classify_pauses_by_words(gaps, sentences) if word_mode else classify_pauses(gaps, sentences, spans)
    pause_rows = []
    if has_speech and span_start > 0:
        pause_rows.append({"pause_class": "leading", "start_s": 0.0, "end_s": span_start, "sent_idx": -1})
    for (g0, g1), (cls, owner) in zip(gaps, classes):
        pause_rows.append({"pause_class": cls, "start_s": g0, "end_s": g1, "sent_idx": owner})
    if has_speech and clip["audio_duration_s"] - span_end > 0:
        pause_rows.append({"pause_class": "trailing", "start_s": span_end, "end_s": clip["audio_duration_s"], "sent_idx": -1})
    for k, p in enumerate(pause_rows):
        p.update({"id": clip["id"], "pause_idx": k, "duration_s": p["end_s"] - p["start_s"],
                  "counted": (p["end_s"] - p["start_s"]) >= MIN_PAUSE_S})
    mids = [p for p in pause_rows if p["pause_class"] == "mid_sentence" and p["counted"]]

    # ---- sentences
    sent_rows = []
    for s, pieces, (sp0, sp1) in zip(sentences, pieces_of, spans):
        speech_time = float((pieces[:, 1] - pieces[:, 0]).sum()) if len(pieces) else 0.0
        dur = sp1 - sp0
        long_enough = dur >= MIN_RATE_SENTENCE_S and s["n_words"] >= MIN_RATE_WORDS
        artic = s["n_syllables"] / speech_time if speech_time > 0 else np.nan
        # Physically implausible articulation means the timestamps do not match the text (seen on garbled
        # ASR lines); the rate is then not a measurement of the speaker and is withheld.
        suspect = bool(long_enough and (not np.isfinite(artic) or artic > MAX_PLAUSIBLE_SPS))
        rate_ok = long_enough and not suspect
        in_sent_p = (pt >= sp0) & (pt <= sp1)
        in_sent_i = (it >= sp0) & (it <= sp1) & int_ok
        st = st_all[in_sent_p & voiced]
        p_std, p_rng, p_med = _pitch_stats(st)
        db = idb[in_sent_i]
        i_std = float(np.std(db)) if len(db) >= MIN_INTENSITY_FRAMES else np.nan
        i_mean = float(np.mean(db)) if len(db) >= MIN_INTENSITY_FRAMES else np.nan
        end_drop = np.nan
        if speech_time >= MIN_TRAIL_SENTENCE_SPEECH_S and len(db) >= MIN_INTENSITY_FRAMES:
            tail = idb[in_sent_i & (it >= sp1 - TRAIL_WINDOW_S)]
            if len(tail) >= 20 and np.percentile(tail, TRAIL_PEAK_PCT) >= noise_floor + TRAIL_MIN_ABOVE_NOISE_DB:
                # Syllable-peak level (80th percentile), not the mean: the mean is dominated by the
                # inter-word dips, which says nothing about whether the voice itself got quieter.
                end_drop = float(np.percentile(tail, TRAIL_PEAK_PCT) - np.percentile(db, TRAIL_PEAK_PCT))
        term_slope = np.nan
        # Weak voicing (breath, creak) far below the syllable peaks gives unreliable F0; keep strong frames.
        strong = (int_at_pitch >= np.percentile(db, TRAIL_PEAK_PCT) - TERMINAL_MAX_BELOW_PEAK_DB) if len(db) >= MIN_INTENSITY_FRAMES else np.ones(len(pt), bool)
        t_end = _last_voiced_run_end(in_sent_p & voiced & strong, pt)
        if np.isfinite(t_end):
            sel = in_sent_p & voiced & strong & (pt >= t_end - TERMINAL_WINDOW_S) & (pt <= t_end)
            if sel.sum() >= MIN_TERMINAL_FRAMES:
                term_slope = _terminal_slope(pt[sel], st_all[sel], p_med if np.isfinite(p_med) else np.median(st_all[sel]))
        own_mids = [p for p in mids if p["sent_idx"] == s["sent_idx"]]
        n_speech_frames = int((in_sent_p & pitch_speech).sum())
        sent_rows.append({
            "id": clip["id"], "sent_idx": s["sent_idx"], "n_sentences": len(sentences),
            "ts_start": s["ts_start"], "ts_end": s["ts_end"], "win_start": s["win_start"], "win_end": s["win_end"],
            "speech_start": sp0, "speech_end": sp1, "duration_s": dur, "speech_time_s": speech_time,
            "rate_valid": rate_ok, "timing_suspect": suspect,
            "text": s["text"], "n_words": s["n_words"], "n_syllables": s["n_syllables"],
            "speech_rate_wpm": 60 * s["n_words"] / dur if rate_ok else np.nan,
            "artic_rate_sps": artic if rate_ok else np.nan,
            "artic_rate_wpm": 60 * s["n_words"] / speech_time if rate_ok else np.nan,
            "n_mid_pauses": len(own_mids),
            "max_mid_pause_s": max((p["duration_s"] for p in own_mids), default=0.0),
            "mid_pause_time_s": float(sum(p["duration_s"] for p in own_mids)),
            "pitch_std_st": p_std, "pitch_range_st": p_rng, "f0_median_st": p_med,
            "intensity_std_db": i_std, "intensity_mean_db": i_mean,
            "voiced_frac": len(st) / n_speech_frames if n_speech_frames else np.nan,
            "end_drop_db": end_drop, "terminal_slope_st_s": term_slope,
        })
    sdf = pd.DataFrame(sent_rows)

    # ---- answer level
    words = sum(s["n_words"] for s in sentences)
    sylls = sum(s["n_syllables"] for s in sentences)
    speech_time = float((regions[:, 1] - regions[:, 0]).sum()) if has_speech else 0.0
    answer_artic = sylls / speech_time if speech_time >= MIN_RATE_SENTENCE_S and words else np.nan
    answer_suspect = bool((np.isfinite(answer_artic) and answer_artic > MAX_PLAUSIBLE_SPS)
                          or (len(sdf) and sdf["timing_suspect"].any()))
    answer_rate_ok = np.isfinite(answer_artic) and answer_artic <= MAX_PLAUSIBLE_SPS
    valid = sdf[sdf["rate_valid"]] if len(sdf) else sdf
    rates = valid["speech_rate_wpm"].to_numpy() if len(valid) else np.array([])
    usable = sdf[~sdf["timing_suspect"]] if len(sdf) else sdf

    rushed = np.nan
    if len(sdf) >= 3 and not answer_suspect:
        head, rest = sdf.iloc[:2], sdf.iloc[2:]
        if head["duration_s"].sum() >= MIN_RATE_SENTENCE_S and rest["duration_s"].sum() >= MIN_RATE_SENTENCE_S:
            r_head = head["n_words"].sum() / head["duration_s"].sum()
            r_rest = rest["n_words"].sum() / rest["duration_s"].sum()
            rushed = r_head / r_rest if r_rest > 0 else np.nan

    drift = np.nan
    if has_speech and span >= 3 * MIN_RATE_SENTENCE_S and len(sdf) and not answer_suspect:
        third = span / 3
        starts, ends = sdf["speech_start"].to_numpy(), sdf["speech_end"].to_numpy()
        n_w, durs = sdf["n_words"].to_numpy(float), sdf["duration_s"].to_numpy()

        def words_between(a: float, b: float) -> float:
            overlap = np.clip(np.minimum(b, ends) - np.maximum(a, starts), 0, None)
            return float(np.sum(np.where(durs > 0, n_w * overlap / np.where(durs > 0, durs, 1), 0)))

        w_first = words_between(span_start, span_start + third)
        w_last = words_between(span_end - third, span_end)
        drift = w_last / w_first if w_first > 0 else np.nan

    # pitch over the whole answer (speech span)
    in_span_p = (pt >= span_start) & (pt <= span_end) if has_speech else np.zeros(len(pt), bool)
    st_ans = st_all[in_span_p & voiced]
    a_std, a_rng, a_med = _pitch_stats(st_ans)
    db_ans = idb[int_ok]
    mono = []
    if has_speech and span >= MONO_WINDOW_S:
        t0 = span_start
        while t0 + MONO_WINDOW_S <= span_end + 1e-9:
            w = st_all[(pt >= t0) & (pt < t0 + MONO_WINDOW_S) & voiced]
            if len(w) >= MONO_MIN_VOICED_FRAMES:
                mono.append(np.std(w) < MONO_STD_ST)
            t0 += MONO_HOP_S
    n_speech_frames_ans = int((in_span_p & pitch_speech).sum())
    span_min = span / 60 if has_speech and span > 0 else np.nan
    internal = [p for p in pause_rows if p["pause_class"] not in ("leading", "trailing")]
    seg = {
        "id": clip["id"],
        # conciseness
        "audio_duration_s": clip["audio_duration_s"], "speech_span_s": span, "speech_time_s": speech_time,
        "word_count": words, "syllable_count": sylls, "n_sentences": len(sentences),
        # pace
        "speech_rate_wpm": 60 * words / span if answer_rate_ok and span >= MIN_RATE_SENTENCE_S else np.nan,
        "artic_rate_sps": answer_artic if answer_rate_ok else np.nan,
        "artic_rate_wpm": 60 * words / speech_time if answer_rate_ok else np.nan,
        "sent_rate_mean_wpm": float(np.mean(rates)) if len(rates) else np.nan,
        "sent_rate_max_wpm": float(np.max(rates)) if len(rates) else np.nan,
        "sent_rate_std_wpm": float(np.std(rates, ddof=1)) if len(rates) >= 2 else np.nan,
        "sent_rate_cv": float(np.std(rates, ddof=1) / np.mean(rates)) if len(rates) >= 2 else np.nan,
        "n_rate_sentences": len(rates),
        "rushed_opening_ratio": rushed, "rate_drift_ratio": drift,
        # pausing
        "response_latency_s": span_start,
        "pause_ratio": float((gaps[:, 1] - gaps[:, 0]).sum() / span) if has_speech and span > 0 else np.nan,
        "mid_pause_rate_per_min": len(mids) / span_min if sentences else np.nan,
        "long_mid_pause_rate_per_min": sum(p["duration_s"] >= LONG_PAUSE_S for p in mids) / span_min if sentences else np.nan,
        "mid_pause_max_s": max((p["duration_s"] for p in mids), default=0.0) if sentences and has_speech else np.nan,
        "mid_pause_time_ratio": sum(p["duration_s"] for p in mids) / span if sentences and has_speech else np.nan,
        "between_pause_time_ratio": (sum(p["duration_s"] for p in internal if p["pause_class"] == "between_sentence" and p["counted"])
                                     / span) if sentences and has_speech else np.nan,
        "internal_pause_count": sum(p["counted"] for p in internal),
        # expressiveness / composure / projection / assertiveness (experimental)
        "pitch_std_st": a_std, "pitch_range_st": a_rng, "f0_median_st": a_med,
        "intensity_std_db": float(np.std(db_ans)) if len(db_ans) >= MIN_INTENSITY_FRAMES else np.nan,
        "monotony_frac": float(np.mean(mono)) if mono else np.nan, "n_monotony_windows": len(mono),
        "end_drop_median_db": float(usable["end_drop_db"].median()) if len(usable) and usable["end_drop_db"].notna().any() else np.nan,
        "terminal_slope_median_st_s": float(usable["terminal_slope_st_s"].median()) if len(usable) and usable["terminal_slope_st_s"].notna().any() else np.nan,
        "jitter_local": float(frames["jitter"]), "shimmer_local": float(frames["shimmer"]),
        # diagnostics
        "intensity_mean_db": float(np.mean(db_ans)) if len(db_ans) >= MIN_INTENSITY_FRAMES else np.nan,
        "f0_median_hz": float(np.median(f0[in_span_p & voiced])) if (in_span_p & voiced).sum() >= MIN_VOICED_FRAMES else np.nan,
        "voiced_frac": len(st_ans) / n_speech_frames_ans if n_speech_frames_ans else np.nan,
        "octave_jump_frac": _octave_jump_frac(f0),
        "dropout_frac": float(dropout[int_speech].mean()) if int_speech.any() else np.nan,
        "isolated_blips_removed": n_blips,
        "transcript_timing_suspect": answer_suspect,
        "pitch_floor_hz": settings["pitch_floor_hz"], "pitch_ceiling_hz": settings["pitch_ceiling_hz"],
    }
    return sent_rows, seg, pause_rows


# ============================================================================================
# Report, listening sheet, QC plots
# ============================================================================================
def listening_sheet(clips: pd.DataFrame, sent: pd.DataFrame, seg: pd.DataFrame, pauses: pd.DataFrame,
                    n: int = 5) -> pd.DataFrame:
    info = clips.set_index("id")[["split", "audio_path"]]
    duplicate_ids = set(seg.loc[seg["is_duplicate"], "id"])
    rows = []

    def add(kind, frame, col, ascending, start_col=None, end_col=None, text_col=None):
        frame = frame[~frame["id"].isin(duplicate_ids)]
        pick = frame.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)
        for rank, (_, r) in enumerate(pick.iterrows(), 1):
            rows.append({"check": kind, "rank": rank, "id": r["id"], "split": info.loc[r["id"], "split"],
                         "value": round(float(r[col]), 3), "measure": col,
                         "start_s": round(float(r[start_col]), 1) if start_col else np.nan,
                         "end_s": round(float(r[end_col]), 1) if end_col else np.nan,
                         "text": r[text_col] if text_col else "", "audio_path": info.loc[r["id"], "audio_path"],
                         "heard_as_extreme (y/n)": "", "notes": ""})

    valid = sent[sent["rate_valid"]]
    add("fastest sentence", valid, "speech_rate_wpm", False, "speech_start", "speech_end", "text")
    add("slowest sentence", valid, "speech_rate_wpm", True, "speech_start", "speech_end", "text")
    add("most monotone answer", seg[seg["n_monotony_windows"] >= 3], "monotony_frac", False)
    add("most expressive answer", seg[seg["n_monotony_windows"] >= 3], "pitch_std_st", False)
    add("strongest trailing off", sent, "end_drop_db", True, "speech_start", "speech_end", "text")
    add("longest mid-sentence pause", pauses[pauses["pause_class"] == "mid_sentence"], "duration_s", False, "start_s", "end_s")
    add("strongest rise at sentence end", sent, "terminal_slope_st_s", False, "speech_start", "speech_end", "text")
    return pd.DataFrame(rows)


def qc_plot(clip: pd.Series, regions: np.ndarray, frames: dict, ref_hz: float, sent: pd.DataFrame,
            pauses: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    f0 = frames["f0"].astype(float)
    st = np.where(f0 > 0, 12 * np.log2(np.where(f0 > 0, f0, 1) / ref_hz), np.nan)
    for ax in (ax1, ax2):
        for s, e in regions:
            ax.axvspan(s, e, color="#cfe8cf", alpha=0.6, lw=0)
        for _, p in pauses.iterrows():
            if p["pause_class"] == "mid_sentence" and p["counted"]:
                ax.axvspan(p["start_s"], p["end_s"], color="#f4a3a3", alpha=0.8, lw=0)
            elif p["pause_class"] == "between_sentence" and p["counted"]:
                ax.axvspan(p["start_s"], p["end_s"], color="#a3c4f4", alpha=0.8, lw=0)
        for _, r in sent.iterrows():
            ax.axvline(r["win_start"], color="black", lw=0.8, ls="--")
    ax1.plot(frames["pitch_t"], st, ".", ms=2, color="#333")
    ax1.set_ylabel("F0 (st re. speaker)")
    int_db = frames["int_db"].astype(float)
    drop = int_db <= DROPOUT_DB
    ax2.plot(frames["int_t"], np.where(drop, np.nan, int_db), color="#555", lw=0.8)
    if drop.any():  # digital silence: shown as a rug, excluded from every descriptor
        ax2.plot(frames["int_t"][drop], np.full(int(drop.sum()), np.nanmin(np.where(drop, np.nan, int_db))),
                 "|", color="black", ms=10, label="dropout (digital silence, excluded)")
        ax2.legend(loc="lower right", fontsize=8)
    ax2.set_ylabel("intensity (dB)")
    ax2.set_xlabel("time (s)  green = VAD speech, red = mid-sentence pause, blue = between-sentence pause, dashed = sentence start")
    for _, r in sent.iterrows():
        label = f"{r['speech_rate_wpm']:.0f} wpm" if r["rate_valid"] else "<4 s"
        ax1.text(r["win_start"] + 0.1, ax1.get_ylim()[1] if np.isfinite(np.nanmax(st)) else 0, label,
                 va="top", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def nan_table(sent: pd.DataFrame, seg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = sent[sent["rate_valid"]]
    for key, meta in REGISTRY.items():
        frame = valid if meta["level"] == "sentence" else seg
        if meta["name"] not in frame.columns:
            continue
        rows.append({"descriptor": key, "status": meta["status"], "category": meta["category"],
                     "nan_pct": 100 * frame[meta["name"]].isna().mean()})
    return pd.DataFrame(rows)


def duplicate_label_agreement(dups: pd.DataFrame) -> pd.DataFrame:
    """For duplicate recordings, how well do the independently given RecruitView scores agree?"""
    targets = ["confidence_score", "speaking_skills", "overall_performance"]
    meta = pd.concat([pd.read_csv(ROOT / "Datasets" / "Split Dataset" / f"{s}.csv", dtype={"id": str})[["id", *targets]]
                      for s in SPLITS])
    d = dups[dups["duplicate_group_size"] > 1].merge(meta, on="id")
    first = d[~d["is_duplicate"]].set_index("duplicate_of")
    pairs = d[d["is_duplicate"]].join(first[targets], on="duplicate_of", rsuffix="_first")
    rows = []
    for t in targets:
        rows.append({"target": t, "n_pairs": len(pairs),
                     "spearman_between_copies": pairs[[t, f"{t}_first"]].corr(method="spearman").iloc[0, 1],
                     "mean_abs_diff": (pairs[t] - pairs[f"{t}_first"]).abs().mean(),
                     "target_sd_all_clips": meta[t].std()})
    out = pd.DataFrame(rows)
    out.attrs["n_groups"] = int(d["audio_md5"].nunique())
    out.attrs["n_clips"] = len(d)
    return out


def write_report(clips, sent, seg, pauses, settings, pass1, checks, nan_df, dup_table, args) -> Path:
    L: list[str] = []
    add = L.append
    add("# Stage 1: Delivery descriptor extraction report\n")
    add(f"Generated by `scripts/explainability/extract_delivery_descriptors.py`. {len(clips)} clips "
        f"({', '.join(f'{s}: {(clips.split == s).sum()}' for s in SPLITS)}), {clips.participant.nunique()} speakers, "
        f"{len(sent)} sentences, {len(pauses)} pause events.\n")
    add("**These are measurements only.** No percentiles, speaker baselines or judgements are applied here; "
        "that is Stage 2 (comparison, fitted on train only) and Stage 9 (feedback wording).\n")

    add("## Coverage\n")
    add(f"- Clips with a segment row: {seg.id.nunique()} / {len(clips)}")
    add(f"- Clips without timestamped transcript: {(seg.n_sentences == 0).sum()} (text-based descriptors are NaN)")
    add(f"- Clips without any VAD speech: {seg.speech_span_s.isna().sum()}")
    add(f"- Sentences >= {MIN_RATE_SENTENCE_S:.0f} s (eligible for per-sentence rate): {sent.rate_valid.sum()} / {len(sent)} "
        f"({100 * sent.rate_valid.mean():.0f}%)")
    add(f"- Speakers using speaker-specific pitch range: {(settings.source == 'speaker').sum()} / {len(settings)}\n")

    add("## NaN rates\n")
    add("Sentence-level descriptors are counted over sentences >= 4 s; answer-level over all clips. "
        "Some answer-level descriptors are NaN **by definition** when the answer is too short or has too few "
        "sentences (e.g. rate variability needs two sentences >= 4 s); those are marked *structural*.\n")
    structural = {"segment.sent_rate_std_wpm", "segment.sent_rate_cv", "segment.rushed_opening_ratio",
                  "segment.rate_drift_ratio", "segment.monotony_frac", "segment.sent_rate_mean_wpm",
                  "segment.sent_rate_max_wpm"}
    add("| Descriptor | Status | Category | NaN % | < 5%? |")
    add("|---|---|---|---|---|")
    for _, r in nan_df.iterrows():
        ok = "yes" if r.nan_pct < 5 else ("structural" if r.descriptor in structural else "**no**")
        add(f"| `{r.descriptor}` | {r.status} | {r.category} | {r.nan_pct:.1f} | {ok} |")
    add("")

    add("## Sanity checks (automatic)\n")
    for line in checks:
        add(f"- {line}")
    add("")

    add("## Pause classes\n")
    counted = pauses[pauses.counted]
    add(f"Pauses >= {MIN_PAUSE_S} s by class (mid vs between is provisional: 1-s timestamps, +-{BOUNDARY_TOL_S} s tolerance):\n")
    add("| Class | count | median duration (s) | p90 duration (s) |")
    add("|---|---|---|---|")
    for cls, g in counted.groupby("pause_class"):
        add(f"| {cls} | {len(g)} | {g.duration_s.median():.2f} | {g.duration_s.quantile(0.9):.2f} |")
    add("")

    add("## Descriptor summary (train split, answer level)\n")
    train_seg = seg[seg.id.isin(clips.loc[clips.split == "train", "id"])]
    add("| Descriptor | Status | p10 | median | p90 |")
    add("|---|---|---|---|---|")
    for key, meta in REGISTRY.items():
        if meta["level"] != "segment" or meta["name"] not in train_seg or meta["unit"] == "bool":
            continue
        v = train_seg[meta["name"]].dropna()
        if len(v):
            add(f"| `{meta['name']}` ({meta['unit']}) | {meta['status']} | {v.quantile(.1):.3g} | {v.median():.3g} | {v.quantile(.9):.3g} |")
    add("")

    add("## Duplicate recordings in RecruitView\n")
    add(f"{dup_table.attrs['n_clips']} clips form {dup_table.attrs['n_groups']} groups of **byte-identical audio** "
        "(always the same participant and question; the video files are sometimes different encodings). They were "
        "rated independently. Descriptors are identical within a group, so Stage 2 must fit reference distributions "
        "on unique recordings (`is_duplicate == False`). The scores given to the copies show how noisy the labels are:\n")
    add("| Target | pairs | Spearman between copies | mean abs. difference | SD over all clips |")
    add("|---|---|---|---|---|")
    for _, r in dup_table.iterrows():
        add(f"| {r.target} | {r.n_pairs} | {r.spearman_between_copies:.2f} | {r.mean_abs_diff:.2f} | {r.target_sd_all_clips:.2f} |")
    add("")

    add("## Listening check (manual, still to do)\n")
    add("`listening_check/listening_check.csv` lists the 5 most extreme sentences/answers for each of: fastest, slowest, most "
        "monotone, most expressive, strongest trailing off, longest mid-sentence pause, strongest final rise. "
        "Listen to each at the given time span and fill in `heard_as_extreme (y/n)`. If the extremes do not "
        "sound extreme, fix the extraction before Stage 2. QC plots for a few of them are in `qc_plots/`.\n")

    add("## Deviations from the stage guide\n")
    add("- **Measurement / comparison / feedback are kept strictly separate.** The guide's step 5 'ratio to the "
        "train median for the same question type' is a comparison against fitted reference data, so it moves to "
        "Stage 2 with the other train-fitted reference tables. Stage 1 outputs raw duration and word count only.")
    add("- **Descriptor status** (active / experimental / diagnostic) is recorded in `descriptor_registry.json`. "
        "Per Stage 0 and review: loudness is diagnostic only; pitch-based measures, trailing off and voice "
        "quality are experimental (measured, normed and validated, but not used for feedback yet).")
    add("- **Pauses are reported as classified rates and ratios**, not raw counts (Stage 0: raw counts track "
        "answer length). The raw count is kept as a diagnostic.")
    add("- **Sentence windows are VAD-trimmed** inside the timestamp window [a, b+1), which gives ~0.1 s "
        "precision for durations instead of 1 s.")
    add("- **Articulation rate** is reported both as syllables/s (standard unit; heuristic syllable counter) "
        "and words/min of speech time (easier to phrase in feedback).")
    add("- Jitter and shimmer (answer level) were added as experimental composure stress proxies for Stage 5.")
    add("- **Fixes found by inspecting the extremes** (first run): (1) isolated VAD blips (< 0.3 s, >= 1 s silence "
        "both sides) are removed, since they stretched sentence spans and turned post-answer silence into fake "
        "mid-sentence pauses; (2) per-sentence rates need >= 5 words, and lines implying > 8 syllables/s (garbled "
        "ASR text in a short window) are marked `timing_suspect` and their rates withheld; (3) digital-silence "
        "frames (recording dropouts, -300 dB) are excluded from all intensity statistics and counted in "
        "`dropout_frac`, and trailing off compares syllable-peak level (p80) rather than mean dB; (4) the terminal F0 slope uses the last voiced run of >= 50 ms, octave-folds frames > 9 st from the sentence median, ignores weak voicing > 25 dB below the syllable peaks "
        "and fits Theil-Sen; (5) a pause is mid-sentence only if it lies inside a sentence's VAD-trimmed speech.")
    add("- **Pitch range rule:** De Looze & Hirst's quartile rule (floor 0.75 x Q1, ceiling 1.5 x Q3), bounded to 0.6-1.8 x the speaker median so halving/doubling errors fall outside the range, instead of the "
        "guide's q15/q85, because pass-1 octave errors contaminate q85 for some speakers.")
    path = P.EXTRACTION_REPORT
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def sanity_checks(clips, seg, settings, pass1, speaker_ref) -> list[str]:
    out = []
    # Compare with the existing handcrafted features (independent implementations).
    try:
        old = pd.concat([pd.read_csv(P.HANDCRAFTED_FEATURES / f"{s}_new_features.csv",
                                     dtype={"id": str}) for s in SPLITS])
        vad_old = pd.concat([pd.read_csv(VAD_DIR / f"{s}_pause_features.csv", dtype={"id": str}) for s in SPLITS])
        m = seg.merge(old[["id", "new_transcript_speech_rate_wpm", "new_relative_pitch_std_semitones", "new_f0_mean_hz"]], on="id") \
               .merge(vad_old[["id", "internal_pause_ratio"]].rename(columns={"internal_pause_ratio": "old_pause_ratio"}), on="id")
        rho = lambda a, b: m[[a, b]].corr(method="spearman").iloc[0, 1]
        out.append(f"Answer speech rate vs existing transcript WPM (file-duration based): Spearman "
                   f"{rho('speech_rate_wpm', 'new_transcript_speech_rate_wpm'):.3f} (expected high; ours excludes leading/trailing silence).")
        out.append(f"Pause ratio (internal gaps / speech span) vs existing VAD internal pause ratio (internal gaps / file duration): Spearman {rho('pause_ratio', 'old_pause_ratio'):.3f}.")
        low = m["f0_median_hz"] < 165
        out.append(f"Answer pitch std (st, Praat, speaker range) vs legacy relative pitch std (st, home-made autocorrelation "
                   f"tracker in compare_handcrafted_improvements.py): Spearman {rho('pitch_std_st', 'new_relative_pitch_std_semitones'):.3f}. "
                   f"For low voices (Praat median < 165 Hz, n={int(low.sum())}) the legacy mean F0 is "
                   f"{m.loc[low, 'new_f0_mean_hz'].median():.0f} Hz vs Praat {m.loc[low, 'f0_median_hz'].median():.0f} Hz: the legacy "
                   "tracker makes octave (doubling) errors on low voices, which inflates its pitch variability. "
                   "**Legacy pitch features (and Stage 0's pitch conclusions) should be treated as unreliable;** "
                   "Stage 3 re-tests pitch with these Praat descriptors.")
    except FileNotFoundError:
        out.append("Existing feature tables not found; cross-implementation checks skipped.")
    # Pitch tracking quality: pass 1 (generic range) vs pass 2 (speaker range).
    p1 = np.array([_octave_jump_frac(pass1[i]) for i in seg["id"]])
    out.append(f"Octave-jump rate (adjacent voiced frames > {OCTAVE_JUMP_ST:.0f} st apart), median over clips: "
               f"pass 1 (75-600 Hz) {np.nanmedian(p1) * 100:.2f}% -> pass 2 (speaker range) "
               f"{np.nanmedian(seg['octave_jump_frac']) * 100:.2f}%; mean {np.nanmean(p1) * 100:.2f}% -> "
               f"{np.nanmean(seg['octave_jump_frac']) * 100:.2f}%.")
    ref = pd.Series(speaker_ref)
    out.append(f"Speaker reference F0: {(ref < 165).sum()} speakers < 165 Hz (median {ref[ref < 165].median():.0f} Hz), "
               f"{(ref >= 165).sum()} >= 165 Hz (median {ref[ref >= 165].median():.0f} Hz). A bimodal split is expected "
               "for adult voices.")
    out.append(f"Pitch floor range used: {settings.pitch_floor_hz.min():.0f}-{settings.pitch_floor_hz.max():.0f} Hz; "
               f"ceiling {settings.pitch_ceiling_hz.min():.0f}-{settings.pitch_ceiling_hz.max():.0f} Hz.")
    train = seg[seg.id.isin(clips.loc[clips.split == "train", "id"])]
    out.append(f"Train answer speech rate median {train.speech_rate_wpm.median():.0f} WPM, articulation "
               f"{train.artic_rate_sps.median():.2f} syll/s (literature: ~120-160 WPM conversational, ~4-5 syll/s "
               "articulation for English; Indian English is often faster).")
    return out


# ============================================================================================
# Main
# ============================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="smoke test on ~N clips")
    args = parser.parse_args()
    for d in (P.S1_TABLES, P.LISTENING, P.QC_PLOTS):
        d.mkdir(parents=True, exist_ok=True)


    clips = load_clips(args.limit)
    print(f"{len(clips)} clips, {clips.participant.nunique()} speakers")
    vad, blips = load_vad(set(clips["id"]))
    print("Hashing audio for duplicate recordings...")
    dups = duplicate_groups(clips)

    print("Praat pass 1 (75-600 Hz)...")
    pass1 = run_parallel(_pass1, list(zip(clips["id"], clips["audio_path"])), args.workers, "pass1")
    settings = speaker_settings(clips, pass1)
    sett = settings.set_index("participant")

    print("Praat pass 2 (speaker-specific range)...")
    jobs = [(r.id, r.audio_path, sett.loc[r.participant, "pitch_floor_hz"], sett.loc[r.participant, "pitch_ceiling_hz"])
            for r in clips.itertuples()]
    frames = run_parallel(_pass2, jobs, args.workers, "pass2")

    # Speaker reference F0 = median of all of the speaker's pass-2 voiced frames.
    speaker_ref = {}
    for participant, g in clips.groupby("participant"):
        voiced = np.concatenate([frames[i]["f0"][frames[i]["f0"] > 0] for i in g["id"]])
        speaker_ref[participant] = float(np.median(voiced)) if len(voiced) else np.nan
    settings["reference_f0_hz"] = settings["participant"].map(speaker_ref)
    settings.to_csv(P.PITCH_SETTINGS, index=False)

    print("Computing descriptors...")
    sent_rows, seg_rows, pause_rows = [], [], []
    for clip in clips.itertuples(index=False):
        clip = pd.Series(clip._asdict())
        s, a, p = describe_clip(clip, vad[clip["id"]], frames[clip["id"]], speaker_ref[clip["participant"]],
                                sett.loc[clip["participant"]], blips[clip["id"]])
        sent_rows += s
        seg_rows.append(a)
        pause_rows += p
    keys = clips[["id", "split", "participant", "question_id", "question_type", "duration_bucket", "video_quality"]] \
        .merge(dups[["id", "is_duplicate", "duplicate_of", "duplicate_group_size"]], on="id")
    sent = keys.merge(pd.DataFrame(sent_rows), on="id")
    seg = keys.merge(pd.DataFrame(seg_rows), on="id")
    pauses = keys[["id", "split", "participant"]].merge(pd.DataFrame(pause_rows), on="id")
    sent.to_csv(P.S1_SENTENCES, index=False)
    seg.to_csv(P.S1_SEGMENTS, index=False)
    pauses.to_csv(P.S1_PAUSES, index=False)
    P.S1_REGISTRY.write_text(json.dumps({
        "_layers": {"measurement": "Stage 1 (this registry)", "comparison": "Stage 2: percentiles vs comparable "
                    "train answers + speaker leave-one-out baseline", "feedback": "Stage 9: LLM wording from evidence JSON"},
        "_status": {"active": "may drive flags/feedback after Stage 2-3", "experimental": "measured + normed + "
                    "validated, not used for feedback yet", "diagnostic": "QC/covariate only, never feedback"},
        "descriptors": REGISTRY}, indent=2), encoding="utf-8")

    for old_plot in P.QC_PLOTS.glob("*.png"):
        old_plot.unlink()
    sheet = listening_sheet(clips, sent, seg, pauses)
    sheet.to_csv(P.LISTENING / "listening_check.csv", index=False)
    by_id = clips.set_index("id")
    for check in ["fastest sentence", "slowest sentence", "most monotone answer", "most expressive answer",
                  "longest mid-sentence pause", "strongest trailing off", "strongest rise at sentence end"]:
        row = sheet[sheet["check"] == check]
        if len(row):
            cid = row.iloc[0]["id"]
            qc_plot(by_id.loc[cid], vad[cid], frames[cid], speaker_ref[by_id.loc[cid, "participant"]],
                    sent[sent.id == cid], pauses[pauses.id == cid],
                    P.QC_PLOTS / f"{check.replace(' ', '_')}_{cid}.png", f"{check}: clip {cid}")

    checks = sanity_checks(clips, seg, settings, pass1, speaker_ref)
    nan_df = nan_table(sent, seg)
    dup_table = duplicate_label_agreement(dups)
    report = write_report(clips, sent, seg, pauses, settings, pass1, checks, nan_df, dup_table, args)
    print("\n".join(checks))
    print(f"Done. Report: {report}")


if __name__ == "__main__":
    main()
