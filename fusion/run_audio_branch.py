"""Run the audio teammate's audio-diagnostics pipeline without touching their checkout.

Their scripts resolve every path against a module-level ROOT (the repo folder). This runner loads
the scripts unchanged from PE/audio_branch/scripts, points ROOT at a separate workspace
(PE/outputs/audio_branch_run), and builds the inputs their loader expects there:

  Datasets/Split Dataset/{split}.csv, {split}_users.txt   copied from PE/split_dataset
  Datasets/RecruitView Audio/{split}_audio_metadata.csv   manifest pointing at PE/outputs/text/wav
  config.yaml                                             their config.yaml, copied verbatim

The WAVs are reused from the text branch: same ffmpeg settings as their converter
(16 kHz, mono, pcm_s16le). Everything they write lands under the workspace.

Steps:
  python fusion/run_audio_branch.py --step prepare
  python fusion/run_audio_branch.py --step extract --extractors legacy,wavlm,hubert --allow-model-download
  python fusion/run_audio_branch.py --step compare
Add --limit 5 to extract/compare for a smoke run (their smoke mode; no test scoring).
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import wave
from pathlib import Path

import pandas as pd

PE = Path(__file__).resolve().parent.parent
AUDIO_REPO = PE / "audio_branch"
SPLIT_SRC = PE / "split_dataset"
WAV_DIR = PE / "outputs" / "text" / "wav"
WORK = PE / "outputs" / "audio_branch_run"
SPLITS = ("train", "val", "test")


def load_their_module(name: str):
    spec = importlib.util.spec_from_file_location(f"audio_{name}", AUDIO_REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.ROOT = WORK  # every path in their code is ROOT / <config path>
    return mod


def prepare() -> None:
    split_dir = WORK / "Datasets" / "Split Dataset"
    audio_dir = WORK / "Datasets" / "RecruitView Audio"
    split_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(AUDIO_REPO / "config.yaml", WORK / "config.yaml")
    for split in SPLITS:
        shutil.copyfile(SPLIT_SRC / f"{split}.csv", split_dir / f"{split}.csv")
        df = pd.read_csv(SPLIT_SRC / f"{split}.csv", dtype={"id": str, "user_no": str, "question_id": str})
        (split_dir / f"{split}_users.txt").write_text(
            "\n".join(sorted(df.user_no.unique())) + "\n", encoding="utf-8")
        rows = []
        for r in df.itertuples():
            wav = WAV_DIR / f"{r.id}.wav"
            with wave.open(str(wav), "rb") as w:
                assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2), wav
                dur = w.getnframes() / w.getframerate()
            rows.append({"id": r.id, "file_name": r.file_name, "user_no": r.user_no,
                         "question_id": r.question_id, "split": split, "user_id": r.user_no,
                         "original_video_path": r.file_name, "audio_path": str(wav),
                         "duration_seconds": f"{dur:.6f}", "sample_rate": 16000, "channels": 1,
                         "status": "already_existed", "error": ""})
        pd.DataFrame(rows).to_csv(audio_dir / f"{split}_audio_metadata.csv", index=False, encoding="utf-8-sig")
        print(f"{split}: {len(rows)} clips, {df.user_no.nunique()} participants")
    print(f"workspace ready: {WORK}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["prepare", "extract", "compare"])
    ap.add_argument("--extractors", default="legacy,wavlm,hubert")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-model-download", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--trials", type=int, default=12)
    args = ap.parse_args()
    if args.step == "prepare":
        prepare()
        return 0
    config = str(WORK / "config.yaml")
    limit = ["--limit", str(args.limit)] if args.limit else []
    if args.step == "extract":
        mod = load_their_module("extract_audio_features")
        argv = ["--config", config, "--extractors", args.extractors, *limit]
        if args.allow_model_download:
            argv.append("--allow-model-download")
        return mod.main(argv)
    mod = load_their_module("compare_audio_models")
    return mod.main(["--config", config, "--n-jobs", str(args.n_jobs), "--trials", str(args.trials), *limit])


if __name__ == "__main__":
    raise SystemExit(main())
