"""Stage 7, step 2: run the unchanged Stage 1 / 4 / 5 / 6 / 2 pipeline on the MIT answer clips.

Nothing is refitted on MIT: pitch settings are per interview (as for any new user), population percentiles are the
RecruitView train references (percentile_tables.json), and the speaker baseline is the same interview's other
answers ("participant" = interview id here; `person` keeps the student id for pre / post pairing).

Per clip, in order (all steps cached by clip id; RecruitView tables are not touched):
  VAD        Silero VAD with the RecruitView settings (scripts/legacy/run_vad_pause.py)
  words      the interview's verbatim Whisper words that fall inside the clip and are not aligned to the
             interviewer (mit_segment_answers.py alignment), shifted to clip time
  Stage 1+4  Praat passes, ASR sentences, describe_clip in word mode (as extract_word_level_descriptors.py)
  Stage 5    audeering SER windows + speaker-relative answer measures (as extract_ser_descriptors.py)
  Stage 6    SQUIM + signal checks, acoustic gate with the RecruitView train thresholds
  Stage 2    category indices / classes and flags with the RecruitView reference set

Outputs  stage7_mit_validation/tables/mit_{segments,sentences,pauses,words,profile,flags}.csv
Run:     python scripts/explainability/mit_run_pipeline.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import extract_delivery_descriptors as X  # noqa: E402
import extract_word_level_descriptors as W  # noqa: E402
import extract_ser_descriptors as ES  # noqa: E402
import run_verbatim_asr as A  # noqa: E402
import run_quality_scores as Q  # noqa: E402
import mit_segment_answers as M  # noqa: E402
import delivery_reference as dr  # noqa: E402
import build_evidence as BE  # noqa: E402

QTYPE = {"Q1": "intro", "Q2": "behavioural", "Q3": "behavioural", "Q4": "self_assessment", "Q5": "self_assessment"}
VAD_OPTIONS = {"threshold": 0.5, "min_speech_duration_ms": 100, "min_silence_duration_ms": 100, "speech_pad_ms": 30}
T = P.S7_TABLES


def load_clips() -> pd.DataFrame:
    a = pd.read_csv(T / "mit_answers.csv", dtype={"id": str})
    return pd.DataFrame({
        "id": a["id"], "split": "mit", "participant": a["interview"], "person": a["participant"],
        "session": a["session"], "question_id": a["question"], "question_type": a["question"].map(QTYPE),
        "duration_bucket": a["audio_duration_s"].map(dr.duration_bucket), "video_quality": "n/a",
        "audio_path": a["audio_path"], "audio_duration_s": a["audio_duration_s"].astype(float),
        "clip_start_s": a["clip_start_s"], "clip_end_s": a["clip_end_s"], "transcript": ""})


def vad_regions(clips: pd.DataFrame) -> tuple[dict, dict]:
    cache_dir = P.CACHE_MIT / "vad"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, blips, model = {}, {}, None
    import soundfile as sf
    for cid, path in zip(clips["id"], clips["audio_path"]):
        f = cache_dir / f"{cid}.json"
        if f.exists():
            reg = np.asarray(json.loads(f.read_text()), dtype=float).reshape(-1, 2)
        else:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad
            if model is None:
                model = load_silero_vad(onnx=False).eval()
            x = sf.read(path, dtype="float32")[0]
            ts = get_speech_timestamps(torch.from_numpy(x), model, sampling_rate=16000, return_seconds=True, **VAD_OPTIONS)
            reg = np.array([(max(0.0, float(t["start"])), min(len(x) / 16000, float(t["end"]))) for t in ts
                            if float(t["end"]) > float(t["start"])], dtype=float).reshape(-1, 2)
            f.write_text(json.dumps(reg.tolist()))
        out[cid], blips[cid] = X.remove_isolated_blips(reg)
    return out, blips


def clip_words(clips: pd.DataFrame) -> dict[str, list[dict]]:
    turns = M.load_turns()
    out = {}
    for iid, g in clips.groupby("participant"):
        words = json.loads((P.CACHE_MIT / "asr" / f"{iid}_{M.ASR_MODEL}.json").read_text(encoding="utf-8"))
        speaker, _, _ = M.align(turns[iid], words)
        for cid, a, b in zip(g["id"], g["clip_start_s"], g["clip_end_s"]):
            out[cid] = [{**w, "start": round(w["start"] - a, 3), "end": round(w["end"] - a, 3)}
                        for w, s in zip(words, speaker)
                        if w["start"] >= a and w["end"] <= b and s != "interviewer"]
    return out


def quality(clips: pd.DataFrame, vad: dict) -> pd.DataFrame:
    import soundfile as sf
    import torch
    from torchaudio.pipelines import SQUIM_OBJECTIVE
    cache_dir = P.CACHE_MIT / "quality"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, model = [], None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for cid, path in zip(clips["id"], clips["audio_path"]):
        f = cache_dir / f"{cid}.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
            continue
        if model is None:
            model = SQUIM_OBJECTIVE.get_model().to(device).eval()
        x = sf.read(path, dtype="float32")[0]
        row = {"id": cid, **Q.signal_checks(x, vad[cid])}
        chunks = Q.squim_chunks(x, vad[cid])
        if chunks:
            full = all(b - a == int(Q.CHUNK_S * Q.SR) for a, b in chunks)
            items = [x[a:b] for a, b in chunks] if full else [x]
            outs = []
            with torch.no_grad():  # MIT answers run to 2 min: mini-batches keep SQUIM inside laptop GPU memory
                for i in range(0, len(items), 4):
                    outs.append(np.stack([v.float().cpu().numpy() for v in
                                          model(torch.from_numpy(np.stack(items[i:i + 4])).to(device))]))
            stoi, pesq, sdr = np.concatenate(outs, axis=1)
            row.update(squim_stoi=float(np.median(stoi)), squim_pesq=float(np.median(pesq)), squim_si_sdr=float(np.median(sdr)))
        row = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}
        f.write_text(json.dumps(row))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    clips = load_clips()
    print(f"{len(clips)} MIT answers, {clips['participant'].nunique()} interviews", flush=True)

    vad, blips = vad_regions(clips)
    X.load_vad = lambda ids: ({i: vad[i] for i in ids}, {i: blips[i] for i in ids})  # for run_ser_windows
    words = clip_words(clips)
    wt = A.word_table(words, vad)
    wt.to_csv(T / "mit_words.csv", index=False)

    print("Praat passes...", flush=True)
    pass1 = X.run_parallel(X._pass1, list(zip(clips["id"], clips["audio_path"])), args.workers, "pass1")
    settings = X.speaker_settings(clips, pass1)
    sett = settings.set_index("participant")
    jobs = [(r.id, r.audio_path, sett.loc[r.participant, "pitch_floor_hz"], sett.loc[r.participant, "pitch_ceiling_hz"])
            for r in clips.itertuples()]
    frames = X.run_parallel(X._pass2, jobs, args.workers, "pass2")
    for iid, g in clips.groupby("participant"):
        voiced = np.concatenate([frames[i]["f0"][frames[i]["f0"] > 0] for i in g["id"]])
        sett.loc[iid, "reference_f0_hz"] = float(np.median(voiced)) if len(voiced) else np.nan
    sett.reset_index().to_csv(T / "mit_pitch_settings.csv", index=False)

    print("Descriptors (word mode)...", flush=True)
    by_clip = {k: g for k, g in wt.groupby("id")} if len(wt) else {}
    sent_rows, seg_rows, pause_rows = [], [], []
    for clip in clips.itertuples(index=False):
        clip = pd.Series(clip._asdict())
        cid = clip["id"]
        w = by_clip.get(cid, wt.iloc[:0])
        kept = w[w["in_vad_speech"]]
        sentences = W.build_sentences(kept, clip["audio_duration_s"])
        s, a, p = X.describe_clip(clip, vad[cid], frames[cid], sett.loc[clip["participant"], "reference_f0_hz"],
                                  sett.loc[clip["participant"]], blips[cid], sentences=sentences)
        for row, sd in zip(s, sentences):
            row["n_filled_pauses"], row["asr_prob_mean"] = sd["n_filled_pauses"], sd["asr_prob_mean"]
        content = kept[~kept["is_filled_pause"]]
        n_fp, span = int(kept["is_filled_pause"].sum()), a["speech_span_s"]
        a.update({"filled_pause_count": n_fp,
                  "filled_pause_rate_per_min": 60 * n_fp / span if np.isfinite(span) and span > 0 else np.nan,
                  "asr_prob_mean": float(content["prob"].mean()) if len(content) else np.nan,
                  "asr_words_outside_vad": int((~w["in_vad_speech"]).sum()),
                  "asr_repetition_suspect": W.repetition_suspect(kept["word"].astype(str).tolist())})
        if a["asr_repetition_suspect"]:
            for row in s:
                row.update({c: np.nan for c in W.RATE_COLUMNS_SENT})
                row["rate_valid"], row["timing_suspect"] = False, True
            a.update({c: np.nan for c in W.RATE_COLUMNS_SEG})
            a["transcript_timing_suspect"] = True
        sent_rows += s
        seg_rows.append(a)
        pause_rows += p
    keys = clips[["id", "split", "participant", "person", "session", "question_id", "question_type", "duration_bucket",
                  "video_quality"]].assign(is_duplicate=False, duplicate_of=clips["id"], duplicate_group_size=1)
    sent = keys.merge(pd.DataFrame(sent_rows), on="id")
    seg = keys.merge(pd.DataFrame(seg_rows), on="id")
    pauses = keys[["id", "split", "participant"]].merge(pd.DataFrame(pause_rows), on="id")

    print("SER (audeering)...", flush=True)
    import run_ser_windows as RS
    win = RS.run("audeering", clips, 16)
    win["f0_st"] = ES.window_pitch(win, clips, sett, args.workers)
    skeys = clips[["id", "participant"]].assign(duplicate_of=np.nan, group=clips["id"])
    win = win.reset_index(drop=True)
    for col in ("arousal", "dominance", "valence", "f0_st"):
        win[f"{'f0' if col == 'f0_st' else col}_z"] = ES.speaker_z(win, skeys, col)
    win.to_csv(T / "mit_windows_audeering.csv", index=False)
    ser = pd.DataFrame([{"id": cid, **ES.answer_measures(g)} for cid, g in win.groupby("id")])
    seg = seg.merge(ser, on="id", how="left")

    print("Quality...", flush=True)
    q = quality(clips, vad)
    seg = seg.merge(q, on="id", how="left")
    gate = json.loads(P.QUALITY_GATE.read_text())
    pcts = np.vstack([np.interp(seg[c].astype(float), gate["composite_train_quantiles"][c], np.linspace(0, 100, 101))
                      for c in gate["composite_measures"]])
    pcts[~np.isfinite(seg[gate["composite_measures"]].astype(float).to_numpy().T)] = np.nan
    seg["quality_composite"] = np.nanmean(pcts, axis=0)
    seg["acoustic_ok"] = (seg["quality_composite"] >= gate["composite_min"]) & ~(seg["clip_frac"] > gate["clip_frac_max"])

    print("Stage 2 classes and flags (RecruitView references)...", flush=True)
    ref = dr.ReferenceSet.load(P.PERCENTILE_TABLES)
    tables = BE.DeliveryTables(seg, sent, pauses, ref)
    prof = tables.profile().reset_index()
    flag_rows = []
    for cid in seg["id"]:
        for f in dr.all_flags(tables.context(cid), ref):
            flag_rows.append({"id": cid, "type": f["type"], "status": f["status"], "category": f["category"]})
    for name, df in (("segments", seg), ("sentences", sent), ("pauses", pauses), ("profile", prof),
                     ("flags", pd.DataFrame(flag_rows, columns=["id", "type", "status", "category"]))):
        df.to_csv(T / f"mit_{name}.csv", index=False)
    print(f"Done: {len(seg)} answers; acoustic gate fails {1 - seg['acoustic_ok'].mean():.1%}; "
          f"{len(flag_rows)} flags. Tables in {T}")


if __name__ == "__main__":
    main()
