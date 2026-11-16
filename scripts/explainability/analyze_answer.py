"""Phase 4, step 1: one answer recording (plus the question text) in, evidence JSON out. No dataset tables are needed;
only stage8_detectors/level_cuts.json (the reference cuts), stage6_quality/quality_gate.json and
stage8_detectors/tables/validation_mit.csv (which dimensions may reach the feedback).

Every measurement is made exactly as for the reference answers, by the same code where it can be imported:
  audio       any file soundfile (or ffmpeg) can read -> 16-kHz mono (polyphase resampling)
  VAD         Silero, the RecruitView settings; isolated blips removed (extract_delivery_descriptors)
  ASR         faster-whisper medium.en, the Stage 4 settings (filler prompt, beam 5, temperature 0)
  pauses      ASR sentences + VAD gaps -> long (>= 1 s) mid-sentence pauses per minute (describe_clip, word mode)
  pitch       Praat, two passes with the speaker's pitch range (De Looze & Hirst rule over all the speaker's answers)
  emotion     audeering dimensional model on 4-s windows (2-s hop); activation = mean(arousal, dominance) over the
              windows that are >= 50% speech
  disfluency  vocametrix classifier and Vox-Profile speech-flow model on 3-s windows (1.5-s hop, >= 50% speech)
  quality     SQUIM + SNR composite, acoustic gate with the RecruitView train thresholds (Stage 6)
  levels      percentiles against the chosen setting's reference (level_cuts.json), weighted index, cuts 20 / 80

Speaker context. Two things depend on the speaker rather than the answer: the Praat pitch range, and the speaker's
typical pitch used to adjust expressiveness for voice pitch. analyze_interview.py takes them from all answers of
the interview (as for MIT). For one clip on its own they come from that clip alone, which is noisier.

Quality gate. Applied (expressiveness and hesitation withheld) in the recorded_answer setting only, as in the
dataset levels. In the live_interview setting it is reported in `internal` but not applied: the gate was fitted on
RecruitView webcam audio, and MIT's far-field audio, on which the levels were validated, fails it almost everywhere.

Models are loaded one at a time and freed (GPU peak about 2 GB, so the 6-GB laptop card is enough).

Run:  python scripts/explainability/analyze_answer.py answer.wav --question "Tell me about yourself." \
          [--setting live_interview|recorded_answer] [--out evidence.json] [--debug]
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from transformers.utils import logging as _hf_logging

_hf_logging.disable_progress_bar()

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import extract_delivery_descriptors as X  # noqa: E402  (Praat settings, speaker pitch range, describe_clip)
import extract_word_level_descriptors as W  # noqa: E402  (sentences from ASR words, repetition loops)
import run_verbatim_asr as A  # noqa: E402  (Whisper prompt, word table)
import run_disfluency_classifier as D  # noqa: E402  (3-s windows)
import run_ser_windows as RS  # noqa: E402  (4-s windows, GPU cap)
import run_quality_scores as Q  # noqa: E402  (signal checks, SQUIM chunks)
import build_answer_evidence as BAE  # noqa: E402  (wording, tips, "where" stretches, usable dimensions)
import fit_answer_levels as FL  # noqa: E402  (dimensions, weights, level names, reasons)

SR = 16000
ASR_MODEL = "medium.en"
VAD_OPTIONS = {"threshold": 0.5, "min_speech_duration_ms": 100, "min_silence_duration_ms": 100, "speech_pad_ms": 30}
AUDIO_DIMENSIONS = ["hesitation", "expressiveness"]   # confidence failed in Phases 1 and 3: never computed here


# ============================================================================================
# Data holders
# ============================================================================================
@dataclass
class Answer:
    id: str
    audio: np.ndarray                 # 16-kHz mono float32
    question: str = ""
    offset_s: float = 0.0             # where the clip starts in the full recording (times in the evidence stay clip-relative)
    speaker_key: str = "speaker"      # answers with the same key share one speaker context
    m: dict = field(default_factory=dict)            # measurements
    win: dict = field(default_factory=dict)          # per-window tables for the "where" stretches

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SR


@dataclass
class Speaker:
    pitch_floor_hz: float
    pitch_ceiling_hz: float
    reference_f0_hz: float            # median voiced F0 over all answers (semitone reference for pitch variability)
    speaker_f0_hz: float              # median of the answers' median F0 (voice-pitch adjustment of expressiveness)
    source: str = "computed"


# ============================================================================================
# Audio
# ============================================================================================
def load_audio(path: str | Path) -> np.ndarray:
    """Any audio (or video) file -> 16-kHz mono float32."""
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    except Exception:  # formats libsndfile cannot read (mp4, m4a, ...): let ffmpeg decode and resample
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                             check=True, capture_output=True).stdout
        return np.frombuffer(raw, np.float32).copy()
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32)


def free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================================
# Steps (each runs over all answers, then frees its model)
# ============================================================================================
def step_vad(answers: list[Answer]) -> None:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    model = load_silero_vad(onnx=False).eval()
    for a in answers:
        ts = get_speech_timestamps(torch.from_numpy(a.audio), model, sampling_rate=SR, return_seconds=True, **VAD_OPTIONS)
        raw = np.array([(max(0.0, float(t["start"])), min(a.duration_s, float(t["end"]))) for t in ts
                        if float(t["end"]) > float(t["start"])], dtype=float).reshape(-1, 2)
        a.m["vad_raw"] = raw                                   # disfluency windows (as the dataset runs)
        a.m["vad"], a.m["n_blips"] = X.remove_isolated_blips(raw)


def step_asr(answers: list[Answer], words: dict[str, list[dict]] | None = None) -> None:
    """Whisper words per answer. `words` (already transcribed, clip-relative times) skips the model."""
    todo = [a for a in answers if not (words and a.id in words)]
    model = None
    if todo:
        from faster_whisper import WhisperModel
        dev = device()
        model = WhisperModel(ASR_MODEL, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    for a in answers:
        if words and a.id in words:
            a.m["words"] = words[a.id]
            continue
        segments, _ = model.transcribe(a.audio, language="en", word_timestamps=True, beam_size=5,
                                       initial_prompt=A.INITIAL_PROMPT, condition_on_previous_text=False,
                                       vad_filter=False, temperature=0.0)
        a.m["words"] = [{"w": w.word.strip(), "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                         "p": round(float(w.probability), 4)} for s in segments for w in (s.words or [])]
    del model
    free_gpu()


def praat_pass1(x: np.ndarray) -> np.ndarray:
    import parselmouth
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    return snd.to_pitch(time_step=X.PITCH_STEP_S, pitch_floor=X.PITCH_PASS1[0],
                        pitch_ceiling=X.PITCH_PASS1[1]).selected_array["frequency"].astype(np.float32)


def praat_pass2(x: np.ndarray, floor: float, ceiling: float) -> dict:
    """Same as extract_delivery_descriptors._pass2, from samples instead of a cached file."""
    import parselmouth
    from parselmouth.praat import call
    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch(time_step=X.PITCH_STEP_S, pitch_floor=floor, pitch_ceiling=ceiling)
    intensity = snd.to_intensity(minimum_pitch=X.INTENSITY_MIN_PITCH, time_step=X.PITCH_STEP_S)
    jitter = shimmer = np.nan
    try:
        points = call(snd, "To PointProcess (periodic, cc)", floor, ceiling)
        jitter = call(points, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, points], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        pass
    return {"pitch_t": pitch.xs().astype(np.float32), "f0": pitch.selected_array["frequency"].astype(np.float32),
            "int_t": intensity.xs().astype(np.float32), "int_db": intensity.values[0].astype(np.float32),
            "jitter": np.float32(jitter), "shimmer": np.float32(shimmer)}


def step_descriptors(answers: list[Answer], speakers: dict[str, Speaker] | None = None) -> dict[str, Speaker]:
    """Praat + ASR sentences -> speech span, long mid-sentence pauses, pitch. The speaker context is computed from
    all answers with the same speaker_key unless given in `speakers`. Returns the context used per key."""
    speakers = dict(speakers or {})
    for key in dict.fromkeys(a.speaker_key for a in answers):
        group = [a for a in answers if a.speaker_key == key]
        given = speakers.get(key)
        if given is None:
            pass1 = {a.id: praat_pass1(a.audio) for a in group}
            sett = X.speaker_settings(pd.DataFrame({"participant": key, "id": [a.id for a in group]}), pass1).iloc[0]
            floor, ceiling = float(sett["pitch_floor_hz"]), float(sett["pitch_ceiling_hz"])
        else:
            floor, ceiling = given.pitch_floor_hz, given.pitch_ceiling_hz
        frames = {a.id: praat_pass2(a.audio, floor, ceiling) for a in group}
        if given is None:
            voiced = np.concatenate([frames[a.id]["f0"][frames[a.id]["f0"] > 0] for a in group])
            ref_hz = float(np.median(voiced)) if len(voiced) else np.nan
        else:
            ref_hz = given.reference_f0_hz
        settings = pd.Series({"pitch_floor_hz": floor, "pitch_ceiling_hz": ceiling})
        for a in group:
            wt = A.word_table({a.id: a.m["words"]}, {a.id: a.m["vad_raw"]})
            if len(wt):
                wt = wt.sort_values(["start", "word_idx"])
            kept = wt[wt["in_vad_speech"]] if len(wt) else wt
            sentences = W.build_sentences(kept, a.duration_s) if len(kept) else []
            clip = pd.Series({"id": a.id, "audio_duration_s": a.duration_s, "transcript": ""})
            _, seg, _ = X.describe_clip(clip, a.m["vad"], frames[a.id], ref_hz, settings, a.m["n_blips"],
                                        sentences=sentences)
            a.m.update({k: seg[k] for k in ("speech_span_s", "long_mid_pause_rate_per_min", "pitch_std_st",
                                            "f0_median_hz")})
            a.m["n_words"] = int((~kept["is_filled_pause"]).sum()) if len(kept) else 0
            a.m["asr_repetition_suspect"] = bool(len(kept)) and W.repetition_suspect(kept["word"].astype(str).tolist())
        if given is None:
            f0s = [a.m["f0_median_hz"] for a in group if np.isfinite(a.m["f0_median_hz"])]
            speakers[key] = Speaker(floor, ceiling, ref_hz, float(np.median(f0s)) if f0s else np.nan,
                                    source=f"computed from {len(group)} answer(s)")
    return speakers


def step_emotion(answers: list[Answer], batch: int = 8) -> None:
    import ser_models as M
    model = M.load("audeering", device())
    for a in answers:
        spans = RS.windows_for(len(a.audio))
        full = [(s, e) for s, e in spans if e - s == int(RS.WIN_S * SR)]
        outs = [M.predict(model, np.stack([a.audio[s:e] for s, e in full[i:i + batch]])) for i in range(0, len(full), batch)]
        outs += [M.predict(model, a.audio[s:e][None, :]) for s, e in spans if e - s != int(RS.WIN_S * SR)]
        out = np.concatenate(outs) if outs else np.empty((0, 3), np.float32)
        w = pd.DataFrame({"start": [s / SR for s, _ in spans], "end": [e / SR for _, e in spans],
                          "speech_frac": [RS.speech_fraction(a.m["vad"], s / SR, e / SR) for s, e in spans],
                          "arousal": out[:, 0] if len(out) else [], "dominance": out[:, 1] if len(out) else []})
        w = w[w["speech_frac"] >= 0.5]
        w["activation"] = (w["arousal"] + w["dominance"]) / 2.0
        a.win["emotion"] = w
        a.m["ser_n_windows"] = len(w)
        a.m["activation"] = float(w["activation"].mean()) if len(w) else np.nan
    del model
    free_gpu()


def _disfluency_chunks(a: Answer) -> tuple[list, list[np.ndarray]]:
    w = D.windows(len(a.audio), a.m["vad_raw"])
    n = int(D.WIN_S * SR)
    chunks = [np.pad(a.audio[int(s * SR):int(e * SR)], (0, max(0, n - (int(e * SR) - int(s * SR)))))[:n] for s, e, _ in w]
    return w, chunks


def step_disfluency(answers: list[Answer], batch: int = 16) -> None:
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    dev = device()
    fe = AutoFeatureExtractor.from_pretrained(P.HF_DISFLUENCY, local_files_only=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(P.HF_DISFLUENCY, local_files_only=True,
                                                              use_safetensors=True).eval().to(dev)
    if dev == "cuda":
        model.half()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    for a in answers:
        w, chunks = _disfluency_chunks(a)
        out = []
        for b in range(0, len(chunks), batch):
            inp = fe(chunks[b:b + batch], sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                logits = model(inp["input_values"].to(dev, dtype=model.dtype)).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        pr = np.concatenate(out) if out else np.empty((0, len(labels)), np.float32)
        t = pd.DataFrame(pr, columns=[f"p_{c}" for c in labels])
        t.insert(0, "start", [s for s, _, _ in w])
        t.insert(1, "end", [e for _, e, _ in w])
        a.win["disfluency"] = t
        a.m["n_windows"] = len(t)
        a.m["p_interjection"] = float(t["p_interjection"].mean()) if len(t) else np.nan
        a.m["p_repetition_block"] = float((t["p_Wordrepetition"] + t["p_block"]).mean()) if len(t) else np.nan
    del model
    free_gpu()


def step_speech_flow(answers: list[Answer], batch: int = 8) -> None:
    import torch
    import voxprofile_models as VP
    model = VP.load("speech_flow", device())
    for a in answers:
        _, chunks = _disfluency_chunks(a)
        x = VP.normalise(np.stack(chunks)) if chunks else np.empty((0, int(D.WIN_S * SR)), np.float32)
        out = []
        for b in range(0, len(x), batch):
            with torch.no_grad():
                fl, _ = model(torch.from_numpy(x[b:b + batch]).to(device()))
            out.append(torch.softmax(fl, -1)[:, 1].float().cpu().numpy())
        p = np.concatenate(out) if out else np.empty(0, np.float32)
        a.win["disfluency"]["p_disfluent"] = p
        a.m["sf_p_disfluent"] = float(p.mean()) if len(p) else np.nan
    del model
    free_gpu()


def step_quality(answers: list[Answer]) -> None:
    import torch
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    dev = device()
    model = SQUIM_OBJECTIVE.get_model().to(dev).eval()
    gate = json.loads(P.QUALITY_GATE.read_text())
    for a in answers:
        row = Q.signal_checks(a.audio, a.m["vad"])
        chunks = Q.squim_chunks(a.audio, a.m["vad"])
        if chunks:
            full = all(e - s == int(Q.CHUNK_S * SR) for s, e in chunks)
            items = [a.audio[s:e] for s, e in chunks] if full else [a.audio]
            outs = []
            with torch.no_grad():
                for i in range(0, len(items), 4):
                    outs.append(np.stack([v.float().cpu().numpy() for v in
                                          model(torch.from_numpy(np.stack(items[i:i + 4])).to(dev))]))
            stoi, pesq, _ = np.concatenate(outs, axis=1)
            row.update(squim_stoi=float(np.median(stoi)), squim_pesq=float(np.median(pesq)))
        pcts = [float(np.interp(row.get(c, np.nan), gate["composite_train_quantiles"][c], np.linspace(0, 100, 101)))
                for c in gate["composite_measures"] if row.get(c) is not None and np.isfinite(row.get(c, np.nan))]
        a.m["quality_composite"] = float(np.mean(pcts)) if pcts else np.nan
        a.m["clip_frac"] = row["clip_frac"]
        a.m["acoustic_ok"] = bool(np.isfinite(a.m["quality_composite"]) and a.m["quality_composite"] >= gate["composite_min"]
                                  and not row["clip_frac"] > gate["clip_frac_max"])
    del model
    free_gpu()


# ============================================================================================
# Levels from level_cuts.json
# ============================================================================================
QS = FL.QS


def pct_from_quantiles(q: list[float], x: float) -> float:
    """Percentile (0-100) of x in a reference given as its 0, 0.5, ..., 100th percentiles. Ties in the reference
    (e.g. 80% of recorded answers have no long mid-sentence pause) get the middle of their range, as the mid-rank
    percentile used for the dataset answers."""
    if x is None or not np.isfinite(x):
        return np.nan
    q = np.asarray(q, float)
    lo, hi = np.searchsorted(q, x, "left"), np.searchsorted(q, x, "right")
    if hi > lo:                                        # x equals one or more reference quantiles
        return float((QS[lo] + QS[hi - 1]) / 2)
    return float(np.interp(x, q, QS))


def load_cuts(setting: str) -> dict:
    cuts = json.loads(P.LEVEL_CUTS.read_text())
    return cuts["settings"][setting or cuts["default_setting"]]


def score(m: dict, speaker: Speaker, cuts: dict) -> dict:
    """The same columns as fit_answer_levels.score() for one answer."""
    adj = cuts["expressiveness_pitch_adjustment"]
    f0 = speaker.speaker_f0_hz if np.isfinite(speaker.speaker_f0_hz) else adj["fill_f0_hz"]
    m = {**m, "speaker_f0_hz": speaker.speaker_f0_hz,
         "activation_adj": m["activation"] - (adj["intercept"] + adj["slope_per_hz"] * f0)}
    out = {"activation_adj": m["activation_adj"]}
    for dim in AUDIO_DIMENSIONS:
        spec = cuts["index"][dim]
        ps = []
        for comp, sign in zip(spec["components"], spec["signs"]):
            p = pct_from_quantiles(cuts["components"][comp], m.get(comp, np.nan))
            p = 100.0 - p if sign < 0 else p
            out[f"pct_{comp}"] = p
            ps.append(p)
        raw = float(np.dot(ps, spec["weights"]))         # NaN if any component is missing
        idx = pct_from_quantiles(spec["raw_mean_quantiles"], raw)
        out[f"{dim}_index"] = idx
        out[f"{dim}_level"] = ("missing" if not np.isfinite(idx) else
                               "low" if idx <= cuts["low"] else "high" if idx >= cuts["high"] else "normal")
    reasons = [FL.REASON_WORDS[c] for c in cuts["index"]["hesitation"]["components"]
               if c in FL.REASON_WORDS and out[f"pct_{c}"] >= cuts["high"]]
    out["hesitation_reasons"] = "; ".join(reasons) if out["hesitation_level"] == "high" else ""
    pp = pct_from_quantiles(cuts["components"]["pitch_std_st"], m.get("pitch_std_st", np.nan))
    out["pct_pitch_std_st"] = pp
    lvl = out["expressiveness_level"]
    out["expressiveness_reasons"] = (FL.PITCH_REASON["low"] if lvl == "low" and pp <= cuts["low"] else
                                     FL.PITCH_REASON["high"] if lvl == "high" and pp >= cuts["high"] else "")
    return out


# ============================================================================================
# Evidence (section 11 format, as build_answer_evidence.py)
# ============================================================================================
def evidence(a: Answer, lv: dict, setting: str, use: dict[str, bool]) -> dict:
    obs, not_assessed = [], []
    gated = setting == "recorded_answer" and not a.m.get("acoustic_ok", True)
    dis = a.win.get("disfluency")
    if dis is not None and len(dis):
        dis = dis.assign(non_fluent=((1.0 - dis["p_fluent"]) + dis["p_disfluent"]) / 2.0)
    for dim in BAE.KEY_TEST:
        if not use.get(dim, False) or dim not in AUDIO_DIMENSIONS:
            not_assessed.append({"dimension": dim, "why": "not validated against human ratings"})
            continue
        if gated:
            not_assessed.append({"dimension": dim, "why": "recording quality too low to judge"})
            continue
        level = lv[f"{dim}_level"]
        if level not in BAE.SAY[dim]:
            continue
        kind, say = BAE.SAY[dim][level]
        o = {"dimension": dim, "level": level, "kind": kind, "say": say,
             "reasons": [r for r in lv[f"{dim}_reasons"].split("; ") if r]}
        where = (BAE.stretch(dis, "non_fluent", highest=level == "high") if dim == "hesitation"
                 else BAE.stretch(a.win.get("emotion"), "activation", highest=level == "high"))
        if where:
            o["where"] = where
        if kind == "issue":
            o["tip"] = BAE.TIPS[dim]
        obs.append(o)
    num = BAE.num
    return {"answer_id": a.id, "question": a.question, "setting": setting, "modality": "audio",
            "observations": obs, "not_assessed": not_assessed,
            "rules": "compared with other candidates' answers only; mention only the observations listed",
            "internal": {"hesitation": num(lv["hesitation_index"]), "expressiveness": num(lv["expressiveness_index"]),
                         "confidence": None, "speech_span_s": num(a.m.get("speech_span_s")),
                         "quality_composite": num(a.m.get("quality_composite")),
                         "acoustic_ok": bool(a.m.get("acoustic_ok", True)), "gate_applied": setting == "recorded_answer",
                         "clip_start_in_recording_s": round(a.offset_s, 2)}}


# ============================================================================================
def measure(answers: list[Answer], words: dict[str, list[dict]] | None = None, log=print) -> None:
    """All model steps (everything except the speaker-dependent pitch and pause descriptors)."""
    t0 = time.time()
    for name, f in [("VAD", lambda: step_vad(answers)), ("ASR", lambda: step_asr(answers, words)),
                    ("emotion model", lambda: step_emotion(answers)),
                    ("disfluency classifier", lambda: step_disfluency(answers)),
                    ("speech-flow model", lambda: step_speech_flow(answers)),
                    ("recording quality", lambda: step_quality(answers))]:
        f()
        log(f"  {name}: done ({time.time() - t0:.0f} s, {len(answers)} answer(s))")


def describe(answers: list[Answer], setting: str | None = None, speakers: dict[str, Speaker] | None = None
             ) -> tuple[list[dict], pd.DataFrame, dict[str, Speaker]]:
    """Pitch / pause descriptors, levels and evidence. Needs measure() first."""
    cuts_all = json.loads(P.LEVEL_CUTS.read_text())
    setting = setting or cuts_all["default_setting"]
    cuts = cuts_all["settings"][setting]
    speakers = step_descriptors(answers, speakers)
    use = BAE.usable_dimensions()
    evs, rows = [], []
    for a in answers:
        sp = speakers[a.speaker_key]
        lv = score(a.m, sp, cuts)
        evs.append(evidence(a, lv, setting, use))
        rows.append({"id": a.id, "setting": setting, "audio_duration_s": a.duration_s,
                     **{k: v for k, v in a.m.items() if np.isscalar(v)}, "speaker_f0_hz": sp.speaker_f0_hz, **lv})
    return evs, pd.DataFrame(rows), speakers


def analyze(answers: list[Answer], setting: str | None = None, speakers: dict[str, Speaker] | None = None,
            words: dict[str, list[dict]] | None = None, log=print) -> tuple[list[dict], pd.DataFrame, dict[str, Speaker]]:
    """Measure, score and describe answers. Returns (evidence per answer, measurement + level table, the speaker
    context used per speaker_key)."""
    measure(answers, words, log)
    return describe(answers, setting, speakers)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("--question", default="")
    ap.add_argument("--setting", choices=["live_interview", "recorded_answer"])
    ap.add_argument("--id")
    ap.add_argument("--out", help="evidence JSON path (default: print)")
    ap.add_argument("--debug", action="store_true", help="also print the measurements and percentiles")
    ap.add_argument("--max-gpu-frac", type=float, default=0.8)
    args = ap.parse_args()
    RS.cap_gpu_memory(args.max_gpu_frac)
    a = Answer(id=args.id or Path(args.audio).stem, audio=load_audio(args.audio), question=args.question)
    evs, table, speakers = analyze([a], args.setting)
    RS.report_gpu_peak()
    text = json.dumps(evs[0], indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.debug:
        print(table.T.to_string())
        print(speakers)


if __name__ == "__main__":
    main()
