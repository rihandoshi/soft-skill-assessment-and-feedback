from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Datasets" / "RecruitView"
SPLIT_DIR = ROOT / "Datasets" / "Split Dataset"
OUTPUT_DIR = ROOT / "Datasets" / "RecruitView Audio"
SPLITS = ("train", "val", "test")


def safe_component(value: object) -> str:
    """Make an identifier safe for a filename on Windows and POSIX."""
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9.-]+", "_", text).strip("._")
    return text or "unknown"


def read_rows(split: str) -> list[dict[str, str]]:
    path = SPLIT_DIR / f"{split}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Split CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"No CSV header found in {path}")
        missing = {"file_name", "user_no", "question_id", "id"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing required mapping columns: {sorted(missing)}")
        return [dict(row) for row in reader]


def resolve_video(row: dict[str, str]) -> Path:

    relative = Path(row["file_name"].replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe video path in file_name: {row['file_name']!r}")
    path = (DATASET_DIR / relative).resolve()
    if not path.is_relative_to(DATASET_DIR.resolve()):
        raise ValueError(f"Video path escapes dataset directory: {row['file_name']!r}")
    return path


def output_path(split: str, row: dict[str, str], video: Path) -> Path:
    # Row id and original video stem together provide stable, traceable names.
    name = "_".join(
        safe_component(value)
        for value in (row["user_no"], row["question_id"], row["id"], video.stem)
    )
    return OUTPUT_DIR / split / f"{name}.wav"


def validate_wav(path: Path) -> tuple[float, int, int]:
    """Return duration, sample rate and channels or raise on invalid WAV."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("WAV file is missing or empty")
    with wave.open(str(path), "rb") as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        frames = wav.getnframes()
        duration = frames / rate if rate else 0.0
        if wav.getcomptype() != "NONE":
            raise ValueError(f"WAV is not uncompressed PCM (compression={wav.getcomptype()})")
        if rate != 16000:
            raise ValueError(f"Expected 16000 Hz, found {rate} Hz")
        if channels != 1:
            raise ValueError(f"Expected mono audio, found {channels} channels")
        if wav.getsampwidth() != 2:
            raise ValueError(f"Expected 16-bit PCM, found {wav.getsampwidth() * 8}-bit")
        if frames <= 0 or duration <= 0:
            raise ValueError("Audio duration must be greater than zero")
    return duration, rate, channels


def extract_one(split: str, row: dict[str, str], ffmpeg: str) -> dict[str, object]:
    video_label = row.get("file_name", "")
    audio = output_path(split, row, Path(video_label.replace("\\", "/")))
    base = {
        **row,
        "split": split,
        "user_id": row.get("user_no", ""),
        "original_video_path": video_label,
        "audio_path": str(audio.relative_to(ROOT)),
    }
    try:
        video = resolve_video(row)
        if not video.is_file():
            raise FileNotFoundError(f"Video does not exist: {video}")
        audio.parent.mkdir(parents=True, exist_ok=True)
        try:
            duration, rate, channels = validate_wav(audio)
            return {**base, "duration_seconds": f"{duration:.6f}", "sample_rate": rate,
                    "channels": channels, "status": "already_existed", "error": ""}
        except (OSError, EOFError, wave.Error, ValueError):
            # Re-extract invalid/incomplete outputs using an atomic replace.
            pass

        fd, temp_name = tempfile.mkstemp(prefix=f".{audio.stem}_", suffix=".wav", dir=audio.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                       "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                       "-c:a", "pcm_s16le", str(temp_path)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                detail = completed.stderr.strip() or f"FFmpeg exited with code {completed.returncode}"
                raise RuntimeError(detail)
            duration, rate, channels = validate_wav(temp_path)
            os.replace(temp_path, audio)
        finally:
            temp_path.unlink(missing_ok=True)
        return {**base, "duration_seconds": f"{duration:.6f}", "sample_rate": rate,
                "channels": channels, "status": "extracted", "error": ""}
    except Exception as exc:  # Keep failures in the per-row manifest and error log.
        return {**base, "duration_seconds": "", "sample_rate": "", "channels": "",
                "status": "failed", "error": str(exc)}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_errors(results_by_split: dict[str, list[dict[str, object]]], selected: set[str]) -> None:
    """Update errors for selected splits while retaining errors in other splits."""
    
    path = OUTPUT_DIR / "extraction_errors.csv"
    errors: dict[tuple[str, str, str], dict[str, object]] = {}
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                key = (row.get("split", ""), row.get("user_id", ""), row.get("question_id", ""))
                errors[key] = row
    for key in [key for key in errors if key[0] in selected]:
        del errors[key]
    for split, rows in results_by_split.items():
        for row in rows:
            if row["status"] == "failed":
                key = (split, str(row["user_id"]), str(row["question_id"]))
                errors[key] = {"split": split, "user_id": row["user_id"],
                               "question_id": row["question_id"],
                               "video_path": row["original_video_path"], "error": row["error"]}
    write_csv(path, list(errors.values()), ["split", "user_id", "question_id", "video_path", "error"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    parser.add_argument("--workers", type=int, default=2,
                        help="Concurrent FFmpeg jobs (default: 2; maximum: 8)")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="FFmpeg executable or path")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    ffmpeg = shutil.which(args.ffmpeg) or (args.ffmpeg if Path(args.ffmpeg).is_file() else None)
    if not ffmpeg:
        print("ERROR: FFmpeg was not found. Install FFmpeg and add it to PATH, or pass --ffmpeg PATH.",
              file=sys.stderr)
        return 2
    selected = list(SPLITS) if args.split == "all" else [args.split]
    try:
        source_rows = {split: read_rows(split) for split in selected}
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    results: dict[str, list[dict[str, object]]] = {}
    for split in selected:
        rows = source_rows[split]
        split_results: list[dict[str, object] | None] = [None] * len(rows)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(extract_one, split, row, ffmpeg): i for i, row in enumerate(rows)}
            for future in as_completed(futures):
                split_results[futures[future]] = future.result()
        results[split] = [result for result in split_results if result is not None]
        source_fields = list(rows[0].keys()) if rows else []
        manifest_fields = list(dict.fromkeys(source_fields + [
            "split", "user_id", "original_video_path", "audio_path", "duration_seconds",
            "sample_rate", "channels", "status", "error",
        ]))
        write_csv(OUTPUT_DIR / f"{split}_audio_metadata.csv", results[split], manifest_fields)
    write_errors(results, set(selected))

    print("RecruitView Audio Extraction\n============================")
    for split in selected:
        rows = results[split]
        extracted = sum(row["status"] == "extracted" for row in rows)
        existing = sum(row["status"] == "already_existed" for row in rows)
        failed = sum(row["status"] == "failed" for row in rows)
        duration = sum(float(row["duration_seconds"] or 0) for row in rows)
        label = "Validation" if split == "val" else split.capitalize()
        print(f"\n{label}:\n  Total samples: {len(rows)}\n  Successfully extracted: {extracted}"
              f"\n  Already existed: {existing}\n  Failed: {failed}"
              f"\n  Total audio duration: {duration / 3600:.2f} hours ({duration:.1f} seconds)")
    total = [row for rows in results.values() for row in rows]
    print(f"\nTotal samples processed: {len(total)}")
    print(f"Total failures: {sum(row['status'] == 'failed' for row in total)}")
    print("\nAll successful audio: WAV, 16 kHz, mono, 16-bit PCM")
    print(f"Metadata and errors: {OUTPUT_DIR}")
    return int(any(row["status"] == "failed" for row in total))


if __name__ == "__main__":
    raise SystemExit(main())
