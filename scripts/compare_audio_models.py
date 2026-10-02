from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import warnings
import wave


'Atomic per-clip cache writes.'


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.stem}_', suffix='.json', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_npz(path: Path, **arrays) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.stem}_', suffix='.npz', dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_csv(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.stem}_', suffix='.csv', dir=path.parent)
    os.close(fd)
    try:
        df.to_csv(name, index=False)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


'Read the central YAML configuration (JSON-subset YAML works without PyYAML).'


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_CONFIG = ROOT / 'config.yaml'


def load_config(path: str | Path | None=None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    raw = config_path.read_text(encoding='utf-8')
    try:
        import yaml
        config = yaml.safe_load(raw)
    except ImportError:
        config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError(f'Configuration must be a mapping: {config_path}')
    config['_config_path'] = str(config_path.resolve())
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


'Authoritative split loading and leakage/mapping checks.'


SPLITS = ('train', 'val', 'test')


TARGET_DEFAULTS = {'confidence': 'confidence_score', 'speaking_skills': 'speaking_skills', 'overall_performance': 'overall_performance'}


def load_split_data(config: dict, limit: int | None=None):
    split_dir = ROOT / config['paths']['split_dir']
    audio_dir = ROOT / config['paths']['audio_dir']
    data = {}
    user_sets = {}
    for split in SPLITS:
        split_csv = split_dir / f'{split}.csv'
        users_path = split_dir / f'{split}_users.txt'
        manifest_path = audio_dir / f'{split}_audio_metadata.csv'
        for required in (split_csv, users_path, manifest_path):
            if not required.is_file():
                raise FileNotFoundError(f'Required fixed split input missing: {required}')
        split_df = pd.read_csv(split_csv, dtype={'id': str, 'user_no': str, 'question_id': str})
        users_txt = {line.strip() for line in users_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()}
        users_csv = set(split_df.user_no.astype(str))
        if users_txt != users_csv:
            raise ValueError(f'Participant-list/CSV mismatch in {split}: users.txt-only={len(users_txt - users_csv)}, CSV-only={len(users_csv - users_txt)}')
        if split_df.id.duplicated().any():
            raise ValueError(f'Duplicate clip ids in authoritative {split}.csv')
        manifest = pd.read_csv(manifest_path, dtype={'id': str, 'user_id': str, 'user_no': str, 'question_id': str})
        if manifest.id.duplicated().any():
            raise ValueError(f'Duplicate clip ids in {manifest_path}')
        expected_ids = set(split_df.id)
        manifest_ids = set(manifest.id)
        if expected_ids != manifest_ids:
            raise ValueError(f'Audio manifest id mismatch in {split}: missing={len(expected_ids - manifest_ids)}, extra={len(manifest_ids - expected_ids)}')
        authoritative = split_df[[c for c in ('id', 'user_no', 'question_id', 'question', *config['targets'].values()) if c in split_df.columns]]
        joined = manifest.merge(authoritative, on='id', how='left', suffixes=('', '_split'), validate='one_to_one')
        for col in ('user_no', 'question_id'):
            split_col = f'{col}_split'
            if col in joined and split_col in joined:
                if not joined[col].astype(str).eq(joined[split_col].astype(str)).all():
                    raise ValueError(f'{col} mismatch between {split_csv.name} and {manifest_path.name}')
        if 'user_no_split' in joined:
            joined['user_no'] = joined.user_no_split
            joined.drop(columns='user_no_split', inplace=True)
        if 'question_id_split' in joined:
            joined['question_id'] = joined.question_id_split
            joined.drop(columns='question_id_split', inplace=True)
        for target in config['targets'].values():
            if f'{target}_split' in joined:
                joined[target] = joined[f'{target}_split']
                joined.drop(columns=f'{target}_split', inplace=True)
        joined['user_id'] = joined.user_no.astype(str)
        joined['split'] = split
        if limit is not None:
            if limit < 1:
                raise ValueError('--limit must be >= 1')
            joined = joined.head(limit).copy()
        data[split] = joined
        user_sets[split] = users_csv
    pairs = (('train', 'val'), ('train', 'test'), ('val', 'test'))
    overlaps = {f'{a}/{b}': sorted(user_sets[a] & user_sets[b]) for a, b in pairs}
    overlaps = {k: v for k, v in overlaps.items() if v}
    if overlaps:
        raise ValueError(f'Participant leakage found in the fixed split files: {overlaps}')
    return data


'Small consistent console logger; tqdm can wrap iterables where installed.'


def get_logger(name: str='audio_pipeline') -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def progress(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


ROOT = Path(__file__).resolve().parents[1]


DATASET_DIR = ROOT / 'Datasets' / 'RecruitView'


SPLIT_DIR = ROOT / 'Datasets' / 'Split Dataset'


OUTPUT_DIR = ROOT / 'Datasets' / 'RecruitView Audio'


SPLITS = ('train', 'val', 'test')


def safe_component(value: object) -> str:
    """Make an identifier safe for a filename on Windows and POSIX."""
    text = str(value).strip()
    text = re.sub('[^A-Za-z0-9.-]+', '_', text).strip('._')
    return text or 'unknown'


def read_rows(split: str) -> list[dict[str, str]]:
    path = SPLIT_DIR / f'{split}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Split CSV not found: {path}')
    with path.open('r', newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f'No CSV header found in {path}')
        missing = {'file_name', 'user_no', 'question_id', 'id'} - set(reader.fieldnames)
        if missing:
            raise ValueError(f'{path} is missing required mapping columns: {sorted(missing)}')
        return [dict(row) for row in reader]


def resolve_video(row: dict[str, str]) -> Path:
    relative = Path(row['file_name'].replace('\\', '/'))
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f"Unsafe video path in file_name: {row['file_name']!r}")
    path = (DATASET_DIR / relative).resolve()
    if not path.is_relative_to(DATASET_DIR.resolve()):
        raise ValueError(f"Video path escapes dataset directory: {row['file_name']!r}")
    return path


def output_path(split: str, row: dict[str, str], video: Path) -> Path:
    name = '_'.join((safe_component(value) for value in (row['user_no'], row['question_id'], row['id'], video.stem)))
    return OUTPUT_DIR / split / f'{name}.wav'


def validate_wav(path: Path) -> tuple[float, int, int]:
    """Return duration, sample rate and channels or raise on invalid WAV."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('WAV file is missing or empty')
    with wave.open(str(path), 'rb') as wav:
        rate, channels = (wav.getframerate(), wav.getnchannels())
        frames = wav.getnframes()
        duration = frames / rate if rate else 0.0
        if wav.getcomptype() != 'NONE':
            raise ValueError(f'WAV is not uncompressed PCM (compression={wav.getcomptype()})')
        if rate != 16000:
            raise ValueError(f'Expected 16000 Hz, found {rate} Hz')
        if channels != 1:
            raise ValueError(f'Expected mono audio, found {channels} channels')
        if wav.getsampwidth() != 2:
            raise ValueError(f'Expected 16-bit PCM, found {wav.getsampwidth() * 8}-bit')
        if frames <= 0 or duration <= 0:
            raise ValueError('Audio duration must be greater than zero')
    return (duration, rate, channels)


def extract_one(split: str, row: dict[str, str], ffmpeg: str) -> dict[str, object]:
    video_label = row.get('file_name', '')
    audio = output_path(split, row, Path(video_label.replace('\\', '/')))
    base = {**row, 'split': split, 'user_id': row.get('user_no', ''), 'original_video_path': video_label, 'audio_path': str(audio.relative_to(ROOT))}
    try:
        video = resolve_video(row)
        if not video.is_file():
            raise FileNotFoundError(f'Video does not exist: {video}')
        audio.parent.mkdir(parents=True, exist_ok=True)
        try:
            duration, rate, channels = validate_wav(audio)
            return {**base, 'duration_seconds': f'{duration:.6f}', 'sample_rate': rate, 'channels': channels, 'status': 'already_existed', 'error': ''}
        except (OSError, EOFError, wave.Error, ValueError):
            pass
        fd, temp_name = tempfile.mkstemp(prefix=f'.{audio.stem}_', suffix='.wav', dir=audio.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video), '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(temp_path)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                detail = completed.stderr.strip() or f'FFmpeg exited with code {completed.returncode}'
                raise RuntimeError(detail)
            duration, rate, channels = validate_wav(temp_path)
            os.replace(temp_path, audio)
        finally:
            temp_path.unlink(missing_ok=True)
        return {**base, 'duration_seconds': f'{duration:.6f}', 'sample_rate': rate, 'channels': channels, 'status': 'extracted', 'error': ''}
    except Exception as exc:
        return {**base, 'duration_seconds': '', 'sample_rate': '', 'channels': '', 'status': 'failed', 'error': str(exc)}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_errors(results_by_split: dict[str, list[dict[str, object]]], selected: set[str]) -> None:
    """Update errors for selected splits while retaining errors in other splits."""
    path = OUTPUT_DIR / 'extraction_errors.csv'
    errors: dict[tuple[str, str, str], dict[str, object]] = {}
    if path.exists():
        with path.open('r', newline='', encoding='utf-8-sig') as stream:
            for row in csv.DictReader(stream):
                key = (row.get('split', ''), row.get('user_id', ''), row.get('question_id', ''))
                errors[key] = row
    for key in [key for key in errors if key[0] in selected]:
        del errors[key]
    for split, rows in results_by_split.items():
        for row in rows:
            if row['status'] == 'failed':
                key = (split, str(row['user_id']), str(row['question_id']))
                errors[key] = {'split': split, 'user_id': row['user_id'], 'question_id': row['question_id'], 'video_path': row['original_video_path'], 'error': row['error']}
    write_csv(path, list(errors.values()), ['split', 'user_id', 'question_id', 'video_path', 'error'])


def main_wav() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=[*SPLITS, 'all'], default='all')
    parser.add_argument('--workers', type=int, default=2, help='Concurrent FFmpeg jobs (default: 2; maximum: 8)')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='FFmpeg executable or path')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('--workers must be between 1 and 8')
    ffmpeg = shutil.which(args.ffmpeg) or (args.ffmpeg if Path(args.ffmpeg).is_file() else None)
    if not ffmpeg:
        print('ERROR: FFmpeg was not found. Install FFmpeg and add it to PATH, or pass --ffmpeg PATH.', file=sys.stderr)
        return 2
    selected = list(SPLITS) if args.split == 'all' else [args.split]
    try:
        source_rows = {split: read_rows(split) for split in selected}
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
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
        manifest_fields = list(dict.fromkeys(source_fields + ['split', 'user_id', 'original_video_path', 'audio_path', 'duration_seconds', 'sample_rate', 'channels', 'status', 'error']))
        write_csv(OUTPUT_DIR / f'{split}_audio_metadata.csv', results[split], manifest_fields)
    write_errors(results, set(selected))
    print('RecruitView Audio Extraction\n============================')
    for split in selected:
        rows = results[split]
        extracted = sum((row['status'] == 'extracted' for row in rows))
        existing = sum((row['status'] == 'already_existed' for row in rows))
        failed = sum((row['status'] == 'failed' for row in rows))
        duration = sum((float(row['duration_seconds'] or 0) for row in rows))
        label = 'Validation' if split == 'val' else split.capitalize()
        print(f'\n{label}:\n  Total samples: {len(rows)}\n  Successfully extracted: {extracted}\n  Already existed: {existing}\n  Failed: {failed}\n  Total audio duration: {duration / 3600:.2f} hours ({duration:.1f} seconds)')
    total = [row for rows in results.values() for row in rows]
    print(f'\nTotal samples processed: {len(total)}')
    print(f"Total failures: {sum((row['status'] == 'failed' for row in total))}")
    print('\nAll successful audio: WAV, 16 kHz, mono, 16-bit PCM')
    print(f'Metadata and errors: {OUTPUT_DIR}')
    return int(any((row['status'] == 'failed' for row in total)))


'Extract handcrafted acoustic features from RecruitView WAV manifests.\n\nCore dependencies: numpy and pandas. Run from any directory with:\n    python Utils/extract_audio_features.py\n\nThe speech/silence detector is an energy-based VAD. Pitch uses FFT\nautocorrelation. Jitter/shimmer columns are frame-level proxies, not calibrated\nclinical voice measurements. Filler counts and transcript-based speaking rates\nare explicitly transcript-derived and can be excluded from the audio-only\nbenchmark with --no-transcript-features.\n'


try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise SystemExit(f'Missing dependency: {exc.name}. Install numpy and pandas in this environment.')


ROOT = Path(__file__).resolve().parents[1]


MANIFEST_DIR = ROOT / 'Datasets' / 'RecruitView Audio'


OUTPUT_DIR = ROOT / 'outputs' / 'audio' / 'features' / 'legacy'


SPLITS = ('train', 'val', 'test')


TARGETS = ('confidence_score', 'speaking_skills', 'overall_performance')


IDENTIFIERS = ('id', 'user_no', 'user_id', 'question_id', 'split', 'file_name', 'original_video_path', 'audio_path', 'status')


FILLERS = re.compile('\\b(?:um+|uh+|erm+|hmm+|mm+|like|you know|i mean)\\b', re.I)


WORD = re.compile("\\b[\\w']+\\b", re.UNICODE)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as src:
        channels, rate, width, frames = (src.getnchannels(), src.getframerate(), src.getsampwidth(), src.getnframes())
        if src.getcomptype() != 'NONE' or width not in (1, 2, 3, 4):
            raise ValueError('Expected uncompressed PCM WAV')
        raw = src.readframes(frames)
    if channels < 1 or rate < 1 or frames < 1:
        raise ValueError('WAV has no audio frames')
    if width == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype='<i4').astype(np.float32) / 2147483648.0
    else:
        octets = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        vals = octets[:, 0].astype(np.int32) | octets[:, 1].astype(np.int32) << 8 | octets[:, 2].astype(np.int32) << 16
        vals = (vals ^ 8388608) - 8388608
        samples = vals.astype(np.float32) / 8388608.0
    if samples.size % channels:
        raise ValueError('WAV sample count is inconsistent with its channel count')
    mono = samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples
    return (mono, rate)


def frames_of(signal: np.ndarray, size: int, hop: int) -> np.ndarray:
    if signal.size < size:
        signal = np.pad(signal, (0, size - signal.size))
    count = 1 + (signal.size - size) // hop
    starts = np.arange(count) * hop
    return signal[starts[:, None] + np.arange(size)[None, :]]


def robust_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if not values.size:
        return {f'{prefix}_{stat}': math.nan for stat in ('mean', 'std', 'median', 'min', 'max', 'range', 'variation')}
    mean = float(np.mean(values))
    return {f'{prefix}_mean': mean, f'{prefix}_std': float(np.std(values)), f'{prefix}_median': float(np.median(values)), f'{prefix}_min': float(np.min(values)), f'{prefix}_max': float(np.max(values)), f'{prefix}_range': float(np.ptp(values)), f'{prefix}_variation': float(np.std(values) / (abs(mean) + 1e-12))}


def get_f0(frames: np.ndarray, rates: np.ndarray, sample_rate: int) -> np.ndarray:
    """Estimate frame F0 with normalized FFT-autocorrelation peak selection."""
    low_lag = max(1, int(sample_rate / 400.0))
    high_lag = min(frames.shape[1] - 2, int(sample_rate / 65.0))
    if high_lag <= low_lag:
        return np.array([], dtype=np.float64)
    window = np.hanning(frames.shape[1]).astype(np.float32)
    nfft = 1 << (2 * frames.shape[1] - 1).bit_length()
    result: list[float] = []
    for frame, rms in zip(frames, rates):
        if rms < 0.008:
            continue
        x = (frame - frame.mean()) * window
        if float(np.dot(x, x)) < 1e-07:
            continue
        spectrum = np.fft.rfft(x, n=nfft)
        ac = np.fft.irfft(spectrum * spectrum.conjugate(), n=nfft)[:high_lag + 1]
        if ac[0] <= 0:
            continue
        corr = ac[low_lag:high_lag + 1] / (ac[0] + 1e-12)
        peaks = np.flatnonzero((corr[1:-1] > corr[:-2]) & (corr[1:-1] >= corr[2:])) + 1
        peaks = peaks[corr[peaks] >= 0.3]
        lag = int(peaks[0] + low_lag) if peaks.size else 0
        if lag:
            result.append(sample_rate / lag)
    return np.asarray(result, dtype=np.float64)


def extract_features(audio_path: Path, transcript: str='', use_transcript: bool=True) -> dict[str, object]:
    signal, sr = read_wav(audio_path)
    duration = signal.size / sr
    frame_size = max(1, int(round(sr * 0.025)))
    hop = max(1, int(round(sr * 0.01)))
    short_frames = frames_of(signal, frame_size, hop)
    rms = np.sqrt(np.mean(short_frames * short_frames, axis=1) + 1e-20)
    noise_floor = float(np.quantile(rms, 0.2))
    threshold = max(0.008, noise_floor * 2.5)
    active = rms >= threshold
    frame_seconds = hop / sr
    speech_seconds = float(active.sum() * frame_seconds)
    silence_seconds = max(0.0, duration - speech_seconds)
    padded = np.r_[True, active, True]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    inactive_runs = [(edges[i + 1] - edges[i]) * frame_seconds for i in range(0, len(edges), 2) if i + 1 < len(edges) and (not active[edges[i]])]
    runs: list[tuple[bool, int, int]] = []
    start = 0
    for i in range(1, active.size + 1):
        if i == active.size or active[i] != active[start]:
            runs.append((bool(active[start]), start, i))
            start = i
    pauses = [(end - begin) * frame_seconds for voiced, begin, end in runs if not voiced and begin > 0 and (end < active.size) and active[begin - 1] and active[end] and ((end - begin) * frame_seconds >= 0.25)]
    pitch_hop = max(1, int(round(sr * 0.03)))
    pitch_frames = frames_of(signal, max(1, int(round(sr * 0.04))), pitch_hop)
    pitch_rms = np.sqrt(np.mean(pitch_frames * pitch_frames, axis=1) + 1e-20)
    f0 = get_f0(pitch_frames, pitch_rms, sr)
    features: dict[str, object] = {'total_duration_seconds': duration, 'speech_duration_seconds': speech_seconds, 'silence_duration_seconds': silence_seconds, 'speech_silence_ratio': speech_seconds / max(silence_seconds, 1e-09), 'speech_fraction': speech_seconds / max(duration, 1e-09), 'pause_count': len(pauses), 'pause_rate_per_minute': len(pauses) / max(duration / 60.0, 1e-09), 'pause_mean_duration_seconds': float(np.mean(pauses)) if pauses else 0.0, 'pause_median_duration_seconds': float(np.median(pauses)) if pauses else 0.0, 'pause_max_duration_seconds': float(np.max(pauses)) if pauses else 0.0, 'sample_rate_hz': sr, 'vad_rms_threshold': threshold}
    features.update(robust_stats(f0, 'f0_hz'))
    features.update(robust_stats(rms[active], 'rms_energy') if active.any() else robust_stats(np.array([]), 'rms_energy'))
    if f0.size > 1:
        periods = 1.0 / f0
        features['jitter_local_proxy_percent'] = float(np.mean(np.abs(np.diff(periods))) / max(float(np.mean(periods)), 1e-12) * 100)
        voiced_amplitudes = pitch_rms[pitch_rms >= max(0.008, float(np.quantile(pitch_rms, 0.2)) * 2.5)]
        if voiced_amplitudes.size > 1:
            db = 20 * np.log10(np.maximum(voiced_amplitudes, 1e-12))
            features['shimmer_local_proxy_db'] = float(np.mean(np.abs(np.diff(db))))
        else:
            features['shimmer_local_proxy_db'] = math.nan
    else:
        features['jitter_local_proxy_percent'] = math.nan
        features['shimmer_local_proxy_db'] = math.nan
    if use_transcript:
        words = WORD.findall(transcript or '')
        fillers = FILLERS.findall(transcript or '')
        features['transcript_word_count'] = len(words)
        features['transcript_filler_count'] = len(fillers)
        features['transcript_words_per_second'] = len(words) / max(speech_seconds, 1e-09)
        features['transcript_words_per_minute'] = len(words) / max(speech_seconds / 60.0, 1e-09)
        features['transcript_filler_rate'] = len(fillers) / max(len(words), 1)
    return features


'OpenSMILE eGeMAPSv02 Functionals extractor (88 named features).'


class EGeMAPSExtractor:
    name = 'egemaps'
    feature_group = 'audio_only'

    def __init__(self, config: dict):
        try:
            import opensmile
        except ImportError as exc:
            raise RuntimeError('eGeMAPS needs the optional opensmile package') from exc
        self.opensmile = opensmile
        self.smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02, feature_level=opensmile.FeatureLevel.Functionals)

    def extract(self, audio_path):
        frame = self.smile.process_file(str(audio_path))
        if frame.empty:
            raise ValueError('OpenSMILE returned an empty eGeMAPS vector')
        values = frame.iloc[0].to_dict()
        if len(values) != 88:
            raise ValueError(f'Expected 88 eGeMAPSv02 Functionals, received {len(values)}')
        return {str(key): float(value) if pd.notna(value) else float('nan') for key, value in values.items()}


'Word-timestamp ASR features and evidence using faster-whisper.'


def _word_norm(text: str) -> str:
    return re.sub("[^\\w']", '', text.casefold(), flags=re.UNICODE)


def derive_timing(words: list[dict], duration: float, filler_terms: list[str], thresholds: list[float]):
    tokens = [w for w in words if w.get('word', '').strip() and w.get('start') is not None and (w.get('end') is not None)]
    total = len(tokens)
    starts = [max(0.0, float(w['start'])) for w in tokens]
    ends = [max(starts[i], float(w['end'])) for i, w in enumerate(tokens)]
    spans = [max(0.0, e - s) for s, e in zip(starts, ends)]
    speech_time = sum(spans)
    gaps = []
    for i in range(1, total):
        gap = max(0.0, starts[i] - ends[i - 1])
        if gap > 0:
            gaps.append({'start': ends[i - 1], 'end': starts[i], 'duration': gap, 'before_word': tokens[i].get('word', '')})
    pause_threshold = min(thresholds) if thresholds else 0.3
    pauses = [gap for gap in gaps if gap['duration'] >= pause_threshold]
    filler_phrases = [tuple((_word_norm(w) for w in term.split())) for term in filler_terms]
    normalized = [_word_norm(item.get('word', '')) for item in tokens]
    filler_positions = []
    for i in range(len(normalized)):
        for phrase in filler_phrases:
            if phrase and tuple(normalized[i:i + len(phrase)]) == phrase:
                filler_positions.append({'term': ' '.join(phrase), 'start': starts[i], 'end': ends[min(i + len(phrase) - 1, total - 1)]})
                break
    repetitions = sum((1 for a, b in zip(normalized, normalized[1:]) if a and a == b))
    long_counts = {f'pause_count_ge_{threshold:g}s': sum((g['duration'] >= threshold for g in pauses)) for threshold in thresholds}
    probs = [float(w['probability']) for w in tokens if w.get('probability') is not None]
    longest = max(gaps, key=lambda gap: gap['duration'], default=None)
    filler_total = len(filler_positions)
    return ({'response_latency_seconds': starts[0] if starts else float('nan'), 'words_per_second_total': total / max(duration, 1e-09), 'words_per_second_speaking': total / max(speech_time, 1e-09), 'articulation_rate_words_per_second': total / max(speech_time, 1e-09), 'word_count': total, 'speaking_time_seconds': speech_time, 'pause_count': len(pauses), 'pauses_per_minute': len(pauses) / max(duration / 60, 1e-09), 'pause_mean_seconds': sum((g['duration'] for g in pauses)) / len(pauses) if pauses else 0.0, 'pause_median_seconds': statistics.median((g['duration'] for g in pauses)) if pauses else 0.0, 'pause_max_seconds': max((g['duration'] for g in pauses), default=0.0), 'pause_fraction': min(1.0, sum((g['duration'] for g in pauses)) / max(duration, 1e-09)), 'pause_before_first_word_seconds': starts[0] if starts else float('nan'), **long_counts, 'filler_count': filler_total, 'filler_rate_per_word': filler_total / max(total, 1), 'immediate_repetition_count': repetitions, 'restart_proxy_count': repetitions, 'word_probability_mean': sum(probs) / len(probs) if probs else float('nan'), 'word_probability_min': min(probs) if probs else float('nan'), 'longest_pause_start_seconds': longest['start'] if longest else float('nan'), 'longest_pause_end_seconds': longest['end'] if longest else float('nan'), 'longest_pause_duration_seconds': longest['duration'] if longest else 0.0, 'filler_positions_json': __import__('json').dumps(filler_positions, ensure_ascii=False)}, {'longest_pause': longest, 'filler_positions': filler_positions, 'repeated_words': [tokens[i].get('word', '') for i in range(1, len(normalized)) if normalized[i] and normalized[i] == normalized[i - 1]], 'word_count': total, 'duration_seconds': duration})


class WhisperTimingExtractor:
    name = 'whisper'
    feature_group = 'asr_derived'

    def __init__(self, config: dict, model_cache, allow_download: bool=False):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError('Whisper timing needs optional faster-whisper') from exc
        options = config['extractors']['whisper']
        model_ref = options.get('model_path') or options.get('model_size', 'small')
        if not allow_download and (not __import__('pathlib').Path(model_ref).exists()):
            raise RuntimeError('Whisper model is not a local path; pass --allow-model-download explicitly or set model_path')
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'
        compute = options.get('compute_type', 'auto')
        if device == 'cpu' and compute == 'auto':
            compute = 'int8'
        self.options = options
        self.device = device
        self.model = WhisperModel(str(model_ref), device=device, compute_type=compute, download_root=str(model_cache))

    def transcribe(self, audio_path):
        options = self.options
        segments, info = self.model.transcribe(str(audio_path), word_timestamps=True, beam_size=int(options.get('beam_size', 5)), language=options.get('language'), vad_filter=bool(options.get('vad_filter', False)))
        output_segments = []
        words = []
        for segment in segments:
            items = []
            for word in getattr(segment, 'words', None) or []:
                item = {'word': word.word, 'start': float(word.start), 'end': float(word.end), 'probability': float(word.probability) if word.probability is not None else None}
                words.append(item)
                items.append(item)
            output_segments.append({'start': float(segment.start), 'end': float(segment.end), 'text': segment.text, 'words': items})
        return {'language': getattr(info, 'language', None), 'segments': output_segments, 'words': words}

    def extract(self, audio_path, duration, cache_path, use_cache=True):
        if use_cache and cache_path.is_file():
            import json
            raw = json.loads(cache_path.read_text(encoding='utf-8'))
        else:
            raw = self.transcribe(audio_path)
            atomic_json(cache_path, raw)
        return (derive_timing(raw.get('words', []), duration, self.options.get('filler_terms', []), self.options.get('long_pause_thresholds_seconds', [0.3, 0.7, 1.5])), raw)


'Layer-wise WavLM/HuBERT mean+std embeddings using Hugging Face Transformers.'


def _slug(value: str) -> str:
    return re.sub('[^A-Za-z0-9._-]+', '_', value).strip('._')


class TransformerEmbeddingExtractor:
    feature_group = 'embedding'

    def __init__(self, name: str, config: dict, model_cache, allow_download: bool=False):
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModel
        except ImportError as exc:
            raise RuntimeError('WavLM/HuBERT embeddings require torch and transformers') from exc
        self.torch = torch
        self.name = name
        self.settings = config['extractors'][name]
        self.model_name = self.settings['model_name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cpu':
            print(f'{name}: CUDA unavailable; using CPU (embedding extraction will be slower)')
        common = {'cache_dir': str(model_cache), 'local_files_only': not allow_download, 'trust_remote_code': bool(self.settings.get('trust_remote_code', False))}
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_name, **common)
        self.model = AutoModel.from_pretrained(self.model_name, **common, output_hidden_states=True)
        self.model.eval().to(self.device)
        self.half = self.device.type == 'cuda'
        if self.half:
            self.model.half()
        self.sample_rate = int(config['audio']['sample_rate'])
        self.chunk_samples = int(float(config['audio']['chunk_seconds']) * self.sample_rate)
        self.overlap_samples = int(float(config['audio']['chunk_overlap_seconds']) * self.sample_rate)
        self.speech_only = bool(self.settings.get('speech_only_pooling', False))

    def extract(self, signal: np.ndarray, sample_rate: int, cache_path, use_cache=True):
        if use_cache and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as cached:
                pooled = cached['pooled'].astype(np.float32)
            return pooled
        if sample_rate != self.sample_rate:
            raise ValueError(f'Expected {self.sample_rate} Hz input, received {sample_rate} Hz')
        if signal.ndim != 1:
            raise ValueError('Expected mono waveform')
        window, overlap = (self.chunk_samples, self.overlap_samples)
        step = max(1, window - overlap)
        starts = [0]
        while starts[-1] + window < len(signal):
            starts.append(starts[-1] + step)
        hidden_chunks: list[list[np.ndarray]] = []
        active_intervals = []
        if self.speech_only:
            frame = int(0.025 * sample_rate)
            hop = int(0.01 * sample_rate)
            if len(signal) < frame:
                padded = np.pad(signal, (0, frame - len(signal)))
                rms = np.array([np.sqrt(np.mean(padded * padded))])
                activity = rms >= max(0.008, np.quantile(rms, 0.2) * 2.5)
                active_intervals = [(0, len(signal), bool(activity[0]))]
            else:
                count = 1 + (len(signal) - frame) // hop
                rms = np.array([np.sqrt(np.mean(signal[i * hop:i * hop + frame] ** 2)) for i in range(count)])
                activity = rms >= max(0.008, np.quantile(rms, 0.2) * 2.5)
                active_intervals = [(i * hop, min(len(signal), i * hop + frame), bool(a)) for i, a in enumerate(activity)]
        with self.torch.no_grad():
            for start in starts:
                stop = min(len(signal), start + window)
                chunk = signal[start:stop].astype(np.float32, copy=False)
                inputs = self.processor(chunk, sampling_rate=sample_rate, return_tensors='pt')
                input_values = inputs['input_values'].to(self.device)
                if self.half:
                    input_values = input_values.half()
                kwargs = {'input_values': input_values, 'output_hidden_states': True}
                if 'attention_mask' in inputs:
                    kwargs['attention_mask'] = inputs['attention_mask'].to(self.device)
                output = self.model(**kwargs)
                hidden = [layer[0].float().cpu().numpy() for layer in output.hidden_states]
                left_trim = int(self.overlap_samples / 2 / max(1, len(chunk)) * hidden[-1].shape[0]) if start else 0
                right_trim = int(self.overlap_samples / 2 / max(1, len(chunk)) * hidden[-1].shape[0]) if stop < len(signal) else 0
                lo = min(left_trim, hidden[-1].shape[0] - 1)
                hi = max(lo + 1, hidden[-1].shape[0] - right_trim)
                hidden_chunks.append([layer[lo:hi] for layer in hidden])
        pooled_layers = []
        for layer_index in range(len(hidden_chunks[0])):
            frames = np.concatenate([chunk[layer_index] for chunk in hidden_chunks], axis=0)
            if self.speech_only and active_intervals:
                times = np.linspace(0, len(signal), len(frames), endpoint=False) + len(signal) / (2 * len(frames))
                indices = np.minimum((times / max(1, len(signal)) * len(active_intervals)).astype(int), len(active_intervals) - 1)
                frame_mask = np.asarray([active_intervals[i][2] for i in indices], dtype=bool)
                if frame_mask.any():
                    frames = frames[frame_mask]
            pooled_layers.append(np.concatenate([frames.mean(axis=0), frames.std(axis=0)]))
        pooled = np.stack(pooled_layers).astype(np.float32)
        atomic_npz(cache_path, pooled=pooled, model_name=np.asarray(self.model_name), pooling=np.asarray('mean+std'), layer_count=np.asarray(len(pooled)))
        return pooled

    def selected_vector(self, pooled: np.ndarray, layer: int | str='best_cv', average_last_n: int=4):
        if layer == 'last_n':
            return pooled[-max(1, int(average_last_n)):].mean(axis=0)
        if layer in ('best_cv', 'all'):
            return pooled
        index = int(layer)
        if index < 0:
            index += len(pooled)
        if not 0 <= index < len(pooled):
            raise IndexError(f'Layer {layer} out of bounds for {len(pooled)} pooled layers')
        return pooled[index]


'Compatibility adapter for the hand-crafted waveform feature branch.'


class LegacyWaveformExtractor:
    name = 'legacy'
    feature_group = 'audio_only'

    def extract(self, audio_path, transcript='', **kwargs):
        return extract_features(audio_path, transcript='', use_transcript=False)

    @staticmethod
    def load_cache(path):
        with np.load(path, allow_pickle=False) as cache:
            return dict(zip(cache['names'].astype(str).tolist(), cache['values'].astype(float).tolist()))


'Resumable per-clip feature extraction over the fixed RecruitView splits.'


LOG = get_logger('audio_pipeline.extract')


ID_COLUMNS = ['id', 'user_id', 'question_id', 'split']


def _safe(value: object) -> str:
    return re.sub('[^A-Za-z0-9_.-]+', '_', str(value)).strip('._') or 'unknown'


def _audio(path: Path):
    with wave.open(str(path), 'rb') as wav:
        if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError(f'Expected mono 16 kHz 16-bit PCM, got {wav.getframerate()} Hz, {wav.getnchannels()} channel(s), {wav.getsampwidth() * 8}-bit')
        rate = wav.getframerate()
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32) / 32768.0
    return (data, rate, len(data) / rate)


def _targets(row: pd.Series, config: dict):
    return {name: row.get(column) for name, column in config['targets'].items()}


def _get_features(extractor_name, extractor, split, row, config, cache_root, force):
    clip_id = _safe(row.id)
    audio_path = project_path(row.audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    model_key = 'default'
    if extractor_name in ('wavlm', 'hubert'):
        model_key = _safe(config['extractors'][extractor_name]['model_name'])
        cache_path = cache_root / 'embeddings' / extractor_name / model_key / split / f'{clip_id}.npz'
        if cache_path.is_file() and (not force):
            with np.load(cache_path, allow_pickle=False) as cached:
                shape = cached['pooled'].shape
            layers, dimensions = (int(shape[0]), int(shape[1]))
        else:
            signal, sr, _ = _audio(audio_path)
            pooled = extractor.extract(signal, sr, cache_path, use_cache=False)
            layers, dimensions = (int(pooled.shape[0]), int(pooled.shape[1]))
        catalog_ref = cache_path.relative_to(ROOT).as_posix()
        return {'__embedding_cache_path': catalog_ref, '__layer_count': layers, '__dimension': dimensions, '__model_name': extractor.model_name}
    if extractor_name == 'legacy':
        cache_path = cache_root / 'legacy' / split / f'{clip_id}.json'
    elif extractor_name == 'egemaps':
        cache_path = cache_root / 'egemaps' / 'eGeMAPSv02_Functionals' / split / f'{clip_id}.json'
    elif extractor_name == 'whisper':
        model_key = _safe(config['extractors']['whisper'].get('model_path') or config['extractors']['whisper']['model_size'])
        raw_path = cache_root / 'whisper_raw' / model_key / split / f'{clip_id}.json'
        cache_path = cache_root / 'whisper_features' / model_key / split / f'{clip_id}.json'
    else:
        raise ValueError(f'Unknown extractor: {extractor_name}')
    if cache_path.is_file() and (not force):
        return json.loads(cache_path.read_text(encoding='utf-8'))
    if extractor_name == 'legacy':
        values = extractor.extract(audio_path)
    elif extractor_name == 'egemaps':
        values = extractor.extract(audio_path)
    else:
        duration = float(row.get('duration_seconds') or 0)
        if not duration:
            _, _, duration = _audio(audio_path)
        values, raw = extractor.extract(audio_path, duration, raw_path, use_cache=not force)
        atomic_json(raw_path.with_name(raw_path.stem + '_evidence.json'), {**raw, 'evidence': values[1]})
        values = values[0]
    atomic_json(cache_path, values)
    return values


def _load_legacy_table(split: str, config: dict):
    path = ROOT / config['paths']['legacy_features_dir'] / f'{split}_audio_features.csv'
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype={'id': str})
    excluded = set(ID_COLUMNS) | {'user_no', 'file_name', 'original_video_path', 'audio_path', 'status', 'feature_status', 'feature_error', *config['targets'].values()}
    return {str(row['id']): {c: float(row[c]) if pd.notna(row[c]) else float('nan') for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])} for _, row in frame.iterrows()}


def _make_extractor(name, config, allow_download):
    if name == 'legacy':
        return LegacyWaveformExtractor()
    if name == 'egemaps':
        return EGeMAPSExtractor(config)
    if name == 'whisper':
        return WhisperTimingExtractor(config, project_path(config['paths']['cache_dir']) / 'whisper', allow_download)
    if name in ('wavlm', 'hubert'):
        return TransformerEmbeddingExtractor(name, config, project_path(config['paths']['cache_dir']) / 'huggingface', allow_download)
    raise ValueError(f'Unknown extractor {name!r}')


def run_extractors(extractor_names: list[str], config: dict, limit: int | None=None, force: bool=False, allow_download: bool=False):
    data = load_split_data(config, limit=limit)
    cache_root = project_path(config['paths']['cache_dir'])
    smoke = 'smoke' if limit else ''
    root_features = project_path(config['paths']['features_dir'])
    failures: list[dict] = []
    skipped: list[dict] = []
    errors_path = project_path(config['paths']['reports_dir']) / ('smoke' if limit else '') / 'extraction_errors.csv'
    for name in extractor_names:
        if name not in ('legacy', 'egemaps', 'whisper', 'wavlm', 'hubert'):
            raise ValueError(f'Unsupported extractor {name!r}')
        try:
            extractor = _make_extractor(name, config, allow_download)
        except Exception as exc:
            LOG.warning('Skipping %s: %s', name, exc)
            skipped.append({'extractor': name, 'reason': str(exc)})
            continue
        output_group = 'asr_derived' if name == 'whisper' else 'embedding' if name in ('wavlm', 'hubert') else 'audio_only'
        for split, frame in data.items():
            legacy_table = _load_legacy_table(split, config) if name == 'legacy' else {}
            records = []
            vectors = []
            metadata_rows = []
            iterator = progress(frame.iterrows(), total=len(frame), desc=f'{name}/{split}')
            for _, row in iterator:
                base = {c: row.get(c) for c in ID_COLUMNS}
                base.update(_targets(row, config))
                try:
                    if name == 'legacy' and str(row.id) in legacy_table and (not force):
                        values = legacy_table[str(row.id)]
                        legacy_cache = cache_root / 'legacy' / split / f'{_safe(row.id)}.json'
                        if not legacy_cache.exists():
                            atomic_json(legacy_cache, values)
                    else:
                        values = _get_features(name, extractor, split, row, config, cache_root, force)
                    if name in ('wavlm', 'hubert'):
                        vectors.append((row.copy(), values))
                        metadata_rows.append({**base, **values, 'feature_group': output_group, 'feature_status': 'ok', 'feature_error': ''})
                    else:
                        records.append({**base, **values, 'feature_group': output_group, 'feature_status': 'ok', 'feature_error': ''})
                except Exception as exc:
                    detail = f'{type(exc).__name__}: {exc}'
                    LOG.error('%s/%s clip %s: %s', name, split, row.id, detail)
                    failures.append({'extractor': name, 'split': split, 'id': row.id, 'user_id': row.user_id, 'question_id': row.question_id, 'audio_path': row.audio_path, 'error': detail})
                    record = {**base, 'feature_group': output_group, 'feature_status': 'failed', 'feature_error': detail}
                    if name in ('wavlm', 'hubert'):
                        metadata_rows.append(record)
                    else:
                        records.append(record)
            if name in ('wavlm', 'hubert'):
                out_dir = project_path(config['paths']['embeddings_dir']) / ('smoke' if limit else '') / name
                metadata_path = out_dir / f'{split}_embedding_index.csv'
                atomic_csv(pd.DataFrame(metadata_rows), metadata_path)
                successful = [(row, item) for row, item in vectors if '__embedding_cache_path' in item]
                if successful:
                    arrays = []
                    ids, users, questions = ([], [], [])
                    for row, item in vectors:
                        if '__embedding_cache_path' not in item:
                            continue
                        path = ROOT / item['__embedding_cache_path']
                        with np.load(path, allow_pickle=False) as embedded:
                            arrays.append(embedded['pooled'].astype(np.float32))
                        ids.append(str(row.id))
                        users.append(str(row.user_id))
                        questions.append(str(row.question_id))
                    atomic_npz(out_dir / f'{split}_embeddings.npz', ids=np.asarray(ids), user_ids=np.asarray(users), question_ids=np.asarray(questions), pooled=np.stack(arrays), model_name=np.asarray(config['extractors'][name]['model_name']))
                    atomic_json(out_dir / f'{split}_embedding_schema.json', {'schema_version': 1, 'key': 'id', 'split': split, 'model_name': config['extractors'][name]['model_name'], 'pooling': 'per-transformer-layer mean concatenated with standard deviation', 'pooled_shape': list(arrays[0].shape), 'array_shape': [len(arrays), *arrays[0].shape], 'layer_index_convention': '0 is embedding output, 1..N are transformer block outputs', 'speech_only_pooling': bool(config['extractors'][name].get('speech_only_pooling', False)), 'cache_files': 'one npz per clip, namespaced by encoder/checkpoint/split/id'})
            else:
                out_dir = root_features / ('smoke' if limit else '') / name
                atomic_csv(pd.DataFrame(records), out_dir / f'{split}_features.csv')
            LOG.info('%s/%s complete: %d clips, %d failures', name, split, len(frame), sum((e['extractor'] == name and e['split'] == split for e in failures)))
    report_dir = errors_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(pd.DataFrame(failures, columns=['extractor', 'split', 'id', 'user_id', 'question_id', 'audio_path', 'error']), errors_path)
    availability = report_dir / ('smoke_extractor_availability.json' if limit else 'extractor_availability.json')
    availability.write_text(json.dumps({'skipped': skipped, 'requested': extractor_names}, indent=2), encoding='utf-8')
    return {'failures': failures, 'skipped': skipped, 'splits': data}


def extract_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--extractors', default='all', help='Comma-separated: legacy,egemaps,whisper,wavlm,hubert')
    parser.add_argument('--limit', type=int, default=None, help='Smoke mode: first N clips per split')
    parser.add_argument('--force', action='store_true', help='Ignore valid per-clip caches and recompute')
    parser.add_argument('--allow-model-download', action='store_true', help='Explicitly allow Hugging Face/Whisper weight downloads')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    names = ['legacy', 'egemaps', 'whisper', 'wavlm', 'hubert'] if args.extractors == 'all' else args.extractors.split(',')
    result = run_extractors(names, config, args.limit, args.force, args.allow_model_download)
    LOG.info('Skipped optional extractors: %d; per-clip failures: %d', len(result['skipped']), len(result['failures']))
    return int(bool(result['failures']))


'Reusable metrics, leakage-safe pipelines, grouped CV and cluster bootstrap.'


def metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = (y[valid], p[valid])
    if y.size == 0:
        return {key: float('nan') for key in ('mae', 'rmse', 'r2', 'pearson', 'spearman', 'ccc')}
    var_y, var_p = (float(np.var(y)), float(np.var(p)))
    covariance = float(np.mean((y - y.mean()) * (p - p.mean())))
    pearson = covariance / math.sqrt(var_y * var_p) if var_y > 0 and var_p > 0 else float('nan')
    ranks_y = pd.Series(y).rank(method='average').to_numpy()
    ranks_p = pd.Series(p).rank(method='average').to_numpy()
    spearman = float(np.corrcoef(ranks_y, ranks_p)[0, 1]) if np.std(ranks_y) and np.std(ranks_p) else float('nan')
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    return {'mae': float(np.mean(np.abs(y - p))), 'rmse': rmse, 'r2': float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)) if var_y else float('nan'), 'pearson': pearson, 'spearman': spearman, 'ccc': float(2 * covariance / (var_y + var_p + (y.mean() - p.mean()) ** 2)) if var_y + var_p + (y.mean() - p.mean()) ** 2 > 0 else float('nan')}


def estimator(name: str, params: dict, seed: int, n_jobs: int=1, stable_ridge: bool=False):
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    if name == 'Ridge':
        ridge_params = dict(params)
        if stable_ridge:
            ridge_params['solver'] = 'lsqr'
        return Ridge(**ridge_params)
    if name == 'SVR':
        return SVR(kernel='rbf', **params)
    if name == 'ExtraTrees':
        return ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs, **params)
    if name == 'RandomForest':
        return RandomForestRegressor(random_state=seed, n_jobs=n_jobs, **params)
    if name == 'XGBoost':
        from xgboost import XGBRegressor
        options = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'tree_method': 'hist',
                   'random_state': seed, 'n_jobs': n_jobs, **params}
        try:
            import torch
            import xgboost
            if torch.cuda.is_available() and int(xgboost.__version__.split('.')[0]) >= 2:
                options['device'] = 'cuda'
        except (ImportError, ValueError):
            pass
        return XGBRegressor(**options)
    raise ValueError(f'Unsupported estimator: {name}')


def build_pipeline(X: pd.DataFrame, model_name: str, params: dict, seed: int, standardize: bool, question_id: bool=False, pca_components: int | float | None=None, n_jobs: int=1, stable_ridge: bool=False):
    """All data-dependent transforms are fit inside each sklearn CV fold."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.decomposition import PCA
    categories = ['question_id'] if question_id and 'question_id' in X.columns else []
    numeric = [column for column in X.columns if column not in categories]
    num_steps = [('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True))]
    if standardize or pca_components is not None:
        num_steps.append(('scale', StandardScaler()))
    if pca_components is not None:
        num_steps.append(('pca', PCA(n_components=pca_components, random_state=seed)))
    transformer = ColumnTransformer([('numeric', Pipeline(num_steps), numeric)] + ([('question', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categories)] if categories else []), remainder='drop', verbose_feature_names_out=False)
    return Pipeline([('preprocess', transformer), ('model', estimator(model_name, params, seed, n_jobs, stable_ridge))])


def grouped_oof(X, y, groups, model_name, params, folds, seed, standardize, question_id=False, pca_components=None, n_jobs=1, stable_ridge=False):
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=folds)
    predictions = np.full(len(X), np.nan, dtype=float)
    fold_metrics = []
    for tr, va in splitter.split(X, y, groups):
        model = build_pipeline(X.iloc[tr], model_name, params, seed, standardize, question_id, pca_components, n_jobs, stable_ridge)
        model.fit(X.iloc[tr], y.iloc[tr])
        predictions[va] = model.predict(X.iloc[va])
        fold_metrics.append(metrics(y.iloc[va], predictions[va]))
    return (predictions, metrics(y, predictions), fold_metrics)


def bootstrap_by_participant(y, pred, users, replicates=1000, seed=43):
    """Percentile confidence intervals, resampling whole participants."""
    frame = pd.DataFrame({'y': np.asarray(y, float), 'pred': np.asarray(pred, float), 'user': np.asarray(users, str)})
    participants = frame.user.unique()
    if len(participants) < 2:
        return {}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in metrics(frame.y, frame.pred)}
    grouped = {user: part for user, part in frame.groupby('user')}
    for _ in range(replicates):
        sample = rng.choice(participants, len(participants), replace=True)
        boot = pd.concat([grouped[user] for user in sample], ignore_index=True)
        for name, value in metrics(boot.y, boot.pred).items():
            if np.isfinite(value):
                values[name].append(value)
    return {name: {'lower_95': float(np.quantile(items, 0.025)), 'upper_95': float(np.quantile(items, 0.975))} for name, items in values.items() if items}


def candidate_params(model_name, config, seed, max_trials=12, stable_ridge=False):
    settings = config['modeling']
    if model_name == 'Ridge':
        alphas = settings.get('embedding_ridge_alphas', [0.1, 1.0, 10.0, 100.0, 1000.0]) if stable_ridge else settings['ridge_alphas']
        return [{'alpha': float(value)} for value in alphas]
    if model_name == 'SVR':
        choices = list(itertools.product(settings['svr_C'], settings['svr_epsilon'], settings['svr_gamma']))
        rng = np.random.default_rng(seed)
        take = min(max_trials, len(choices))
        indices = rng.choice(len(choices), take, replace=False)
        return [{'C': float(choices[i][0]), 'epsilon': float(choices[i][1]), 'gamma': choices[i][2]} for i in indices]
    if model_name == 'XGBoost':
        rng = np.random.default_rng(seed)
        return [{'n_estimators': int(rng.choice([150, 300, 500])), 'max_depth': int(rng.choice([2, 3, 4, 6])), 'learning_rate': float(rng.choice([0.02, 0.04, 0.07, 0.1])), 'min_child_weight': float(rng.choice([1, 3, 5, 8])), 'subsample': float(rng.choice([0.7, 0.85, 1.0])), 'colsample_bytree': float(rng.choice([0.7, 0.85, 1.0])), 'reg_alpha': float(rng.choice([0, 0.01, 0.1, 1])), 'reg_lambda': float(rng.choice([1, 3, 8])), 'gamma': float(rng.choice([0, 0.1, 0.3]))} for _ in range(max_trials)]
    rng = np.random.default_rng(seed)
    candidates = []
    def choose(options):
        # np.random.choice coerces mixed float/string lists to strings (for
        # example max_features=0.7 became the invalid value '0.7').
        return options[int(rng.integers(len(options)))]
    for _ in range(max_trials):
        candidates.append({'n_estimators': int(choose([200, 400, 700])), 'max_depth': choose([None, 8, 12, 20]), 'min_samples_split': int(choose([2, 4, 8, 12])), 'min_samples_leaf': int(choose([1, 2, 3, 5])), 'max_features': choose([1.0, 0.7, 0.5, 'sqrt'])})
    return candidates


def search_model(X, y, groups, model_name, configs, folds, seed, standardize=True, question_id=False, pca_components=None, n_jobs=1, stable_ridge=False):
    """Tune by participant-grouped CV RMSE; return OOF predictions for the winner."""
    candidates = candidate_params(model_name, configs, seed, int(configs['modeling'].get('tree_trials', 12)), stable_ridge)
    records, best, best_oof = ([], None, None)
    for index, params in enumerate(candidates):
        try:
            oof, score, fold_scores = grouped_oof(X, y, groups, model_name, params, folds, seed, standardize, question_id, pca_components, n_jobs, stable_ridge)
        except Exception as exc:
            records.append({'model': model_name, 'trial': index, 'params': params, 'error': str(exc)})
            continue
        record = {'model': model_name, 'trial': index, 'params': params, **{f'cv_{k}': v for k, v in score.items()}, 'cv_rmse_fold_mean': float(np.mean([m['rmse'] for m in fold_scores])), 'cv_rmse_fold_std': float(np.std([m['rmse'] for m in fold_scores])), 'error': ''}
        records.append(record)
        if best is None or record['cv_rmse_fold_mean'] < best['cv_rmse_fold_mean']:
            best, best_oof = (record, oof)
    return (best, best_oof, records)


'Leakage-safe hidden-layer sweep and last-N layer averaging.'


def variants(layer_count: int, last_n: int):
    options = [(f'layer_{i}', i) for i in range(layer_count)]
    options.append((f'mean_last_{min(last_n, layer_count)}', 'last_n'))
    return options


def layer_sweep(pooled_by_split: dict[str, np.ndarray], metadata_by_split: dict[str, pd.DataFrame], target: str, config: dict, folds: int, seed: int, encoder='embedding'):
    train_meta = metadata_by_split['train'].reset_index(drop=True)
    y = pd.to_numeric(train_meta[target], errors='coerce')
    valid = y.notna().to_numpy()
    y = y[valid].reset_index(drop=True)
    groups = train_meta.loc[valid, 'user_id'].astype(str).to_numpy()
    layers = pooled_by_split['train'].shape[1]
    n_last = int(config['extractors']['wavlm'].get('average_last_n', 4))
    results = []
    representations = {}
    layer_variants = variants(layers, n_last)
    for layer_number, (name, selection) in enumerate(layer_variants, start=1):

        def at_split(split):
            values = pooled_by_split[split]
            return values[:, -n_last:, :].mean(axis=1) if selection == 'last_n' else values[:, int(selection), :]
        Xall = at_split('train')[valid]
        X = pd.DataFrame(Xall, columns=[f'embedding_{i}' for i in range(Xall.shape[1])])
        _assert_no_target_columns(X, config, f'{encoder} layer-sweep training X for {target}')
        best = None
        best_result = None
        best_oof = None
        for alpha in config['modeling'].get('embedding_ridge_alphas', [0.1, 1.0, 10.0, 100.0, 1000.0]):
            oof, cv, fold_scores = grouped_oof(X, y, groups, 'Ridge', {'alpha': float(alpha)}, folds, seed, True, pca_components=config['modeling'].get('embedding_pca_components'), stable_ridge=True)
            fold_rmse = float(np.mean([m['rmse'] for m in fold_scores]))
            if best is None or fold_rmse < best['cv_rmse_fold_mean']:
                best = {'alpha': float(alpha), 'cv': cv, 'cv_rmse_fold_mean': fold_rmse, 'cv_rmse_fold_std': float(np.std([m['rmse'] for m in fold_scores]))}
                best_result, best_oof = (best, oof)
        val_meta = metadata_by_split['val'].reset_index(drop=True)
        vy = pd.to_numeric(val_meta[target], errors='coerce')
        val_mask = vy.notna().to_numpy()
        Xv_all = at_split('val')
        Xv = pd.DataFrame(Xv_all[val_mask], columns=X.columns)
        _assert_no_target_columns(Xv, config, f'{encoder} layer-sweep validation X for {target}')
        model = build_pipeline(X, 'Ridge', {'alpha': best['alpha']}, seed, True, pca_components=config['modeling'].get('embedding_pca_components'), stable_ridge=True)
        model.fit(X, y)
        vp = model.predict(Xv) if len(Xv) else np.asarray([])
        val_metric = metrics(vy[val_mask], vp) if len(vp) else {}
        results.append({'layer': name, 'layer_index': selection, 'alpha': best['alpha'], 'cv_rmse_fold_mean': best['cv_rmse_fold_mean'], 'cv_rmse_fold_std': best['cv_rmse_fold_std'], **{f'cv_{key}': value for key, value in best['cv'].items()}, **{f'val_{key}': value for key, value in val_metric.items()}})
        representations[name] = {'train': at_split('train'), 'val': at_split('val'), 'alpha': best['alpha'], 'model': model, 'oof': best_oof}
        print(f'[{encoder}][{target}] layer {layer_number}/{len(layer_variants)} {name} complete: '
              f'alpha={best["alpha"]:g}, grouped-CV RMSE={best["cv_rmse_fold_mean"]:.4f}', flush=True)
    table = pd.DataFrame(results).sort_values('cv_rmse_fold_mean', ascending=True)
    winner = str(table.iloc[0].layer)
    print(f'[{encoder}][{target}] layer sweep complete: selected {winner}, '
          f'grouped-CV RMSE={table.iloc[0].cv_rmse_fold_mean:.4f}', flush=True)
    return (winner, representations[winner], table)


'Comparable feature/model benchmarks using the fixed participant splits.'


LOG = get_logger('audio_pipeline.compare')


EXCLUDED = {'id', 'user_id', 'user_no', 'question_id', 'split', 'file_name', 'original_video_path', 'audio_path', 'status', 'feature_group', 'feature_status', 'feature_error'}


def _target_feature_names(config):
    """Names used for targets in source labels and in extractor output tables."""
    return {str(name).strip().casefold()
            for name in (*config['targets'].keys(), *config['targets'].values())}


def _assert_no_target_columns(X, config, context):
    leaked = [column for column in X.columns
              if str(column).strip().casefold() in _target_feature_names(config)]
    if leaked:
        raise AssertionError(f'Target leakage in {context}: target column(s) present in X: {leaked}')


def _split_feature_root(config, smoke):
    return project_path(config['paths']['features_dir']) / ('smoke' if smoke else '')


def load_feature_group(name, data, config, smoke=False):
    root = _split_feature_root(config, smoke)
    all_frames = {}
    feature_names = None
    for split in SPLITS:
        directory = root / name
        path = directory / f'{split}_features.csv'
        if name == 'legacy' and (not path.is_file()):
            path = directory / f'{split}_audio_features.csv'
        if not path.is_file():
            return None
        table = pd.read_csv(path, dtype={'id': str})
        if table.id.duplicated().any():
            raise ValueError(f'Duplicate ids in extracted {name} features: {path}')
        target_columns = _target_feature_names(config)
        present_targets = [column for column in table.columns
                           if str(column).strip().casefold() in target_columns]
        if present_targets:
            print(f'[{name}/{split}] excluding target label column(s) from feature table: '
                  f'{present_targets}', flush=True)
        numeric = [column for column in table.columns if column not in EXCLUDED
                   and str(column).strip().casefold() not in target_columns
                   and pd.api.types.is_numeric_dtype(table[column])]
        if not numeric:
            return None
        ids = data[split].id.astype(str).tolist()
        indexed = table.assign(id=table.id.astype(str)).set_index('id')
        missing = set(ids) - set(indexed.index)
        if missing:
            raise ValueError(f'{name}/{split} feature table is missing {len(missing)} fixed split ids')
        all_frames[split] = indexed.reindex(ids)[numeric].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        _assert_no_target_columns(all_frames[split], config, f'{name}/{split} feature table')
        if feature_names is None:
            feature_names = numeric
        elif set(feature_names) != set(numeric):
            raise ValueError(f'Feature columns differ across {name} split tables')
    return all_frames


def load_embeddings(name, data, config, smoke=False, splits=('train', 'val')):
    root = project_path(config['paths']['embeddings_dir']) / ('smoke' if smoke else '') / name
    output = {}
    for split in splits:
        path = root / f'{split}_embeddings.npz'
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            ids = archive['ids'].astype(str).tolist()
            pooled = archive['pooled'].astype(np.float32)
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate ids in {path}')
        positions = {clip_id: i for i, clip_id in enumerate(ids)}
        expected = data[split].id.astype(str).tolist()
        aligned = np.full((len(expected), *pooled.shape[1:]), np.nan, dtype=np.float32)
        for i, clip_id in enumerate(expected):
            if clip_id in positions:
                aligned[i] = pooled[positions[clip_id]]
        output[split] = aligned
    return output


def combine(*frames):
    return {split: pd.concat([frame[split].reset_index(drop=True) for frame in frames], axis=1) for split in SPLITS}


def label_diagnostics(data, config):
    joined = pd.concat(data.values(), ignore_index=True)
    targets = list(config['targets'].values())
    report = {'target_distributions_by_split': {}, 'target_correlations': {}, 'variance_explained_by_group': {}}
    for split, frame in data.items():
        report['target_distributions_by_split'][split] = {}
        for target in targets:
            y = pd.to_numeric(frame[target], errors='coerce').dropna()
            report['target_distributions_by_split'][split][target] = {'n': int(len(y)), 'dtype': str(frame[target].dtype), 'unique_count': int(y.nunique()), 'min': float(y.min()) if len(y) else None, 'max': float(y.max()) if len(y) else None, 'mean': float(y.mean()) if len(y) else None, 'std': float(y.std()) if len(y) > 1 else None, 'frequencies_top_20': {str(k): int(v) for k, v in y.value_counts().head(20).items()}, 'formulation': 'continuous regression; no binning' if y.nunique() > 10 else 'inspect possible ordinal structure'}
    report['target_correlations'] = joined[targets].corr(method='pearson', min_periods=3).to_dict()
    for group_col in ('user_id', 'question_id'):
        report['variance_explained_by_group'][group_col] = {}
        for target in targets:
            frame = joined[[group_col, target]].dropna()
            y = pd.to_numeric(frame[target], errors='coerce')
            grand = y.mean()
            total = float(((y - grand) ** 2).sum())
            between = sum((len(indices) * (y.loc[indices].mean() - grand) ** 2 for _, indices in frame.groupby(group_col).groups.items())) if total else 0.0
            report['variance_explained_by_group'][group_col][target] = {'eta_squared': float(between / total) if total else None, 'n_groups': int(frame[group_col].nunique()), 'n_rows': int(len(frame))}
    return report


def available_models(feature_kind, has_embedding=False):
    # RBF-SVR is prone to a singular kernel system on the high-dimensional
    # transformer vectors. Keep it for the unchanged handcrafted baseline.
    models = ['Ridge'] if has_embedding else ['Ridge', 'SVR']
    if feature_kind == 'handcrafted':
        models.extend(['ExtraTrees', 'RandomForest'])
        try:
            import xgboost
            models.append('XGBoost')
        except ImportError:
            pass
    return models


def _candidate_record(target, feature_set, model_name, best, val_metrics, oof, oof_metrics, feature_cols, question_variant=False):
    return {'target': target, 'feature_set': feature_set, 'model': model_name, 'params': best['params'], 'feature_columns': list(feature_cols), 'question_id_ablation': question_variant, 'cv_metrics': oof_metrics, 'val_metrics': val_metrics, 'oof_predictions': oof, 'cv_record': best, 'validation_selected': False, 'final_test_metrics': None, 'bootstrap_ci': None}


def _layer_df(values, name='embedding'):
    return pd.DataFrame(values, columns=[f'{name}_{i}' for i in range(values.shape[1])])


def _select_encoder_layer(data, pooled, encoder, config, folds, seed, report_dir, target):
    if pooled is None:
        return None
    metadata = {split: data[split][['id', 'user_id', 'question_id', target]].reset_index(drop=True) for split in ('train', 'val')}
    winner, representation, curve = layer_sweep(pooled, metadata, target, config, folds, seed, encoder=encoder)
    report_dir.mkdir(parents=True, exist_ok=True)
    curve.insert(0, 'encoder', encoder)
    curve.insert(0, 'target', target)
    curve.to_csv(report_dir / f'layer_sweep_{encoder}_{target}.csv', index=False)
    LOG.info('%s layer winner for %s by grouped train CV: %s', encoder, target, winner)
    return {'encoder': encoder, 'layer': winner, **representation}


def _feature_sets_for_target(data, legacy, egemaps, whisper, encoder_winners):
    sets = {}
    if legacy is not None:
        sets['A_legacy_handcrafted'] = {'frames': legacy, 'kind': 'handcrafted'}
    if egemaps is not None:
        sets['B_eGeMAPS'] = {'frames': egemaps, 'kind': 'handcrafted'}
    if egemaps is not None and whisper is not None:
        sets['C_eGeMAPS_plus_Whisper'] = {'frames': combine(egemaps, whisper), 'kind': 'handcrafted'}
    for encoder in ('wavlm', 'hubert'):
        winner = encoder_winners.get(encoder)
        if winner is not None:
            frames = {'train': _layer_df(winner['train'], encoder), 'val': _layer_df(winner['val'], encoder)}
            sets['D_WavLM' if encoder == 'wavlm' else 'E_HuBERT'] = {'frames': frames, 'kind': 'embedding', 'encoder': encoder, 'layer': winner['layer'], 'alpha': winner['alpha']}
    if egemaps is not None and whisper is not None:
        for encoder in ('wavlm', 'hubert'):
            winner = encoder_winners.get(encoder)
            if winner is not None:
                fused = {'train': pd.concat([_layer_df(winner['train'], encoder), egemaps['train'], whisper['train']], axis=1), 'val': pd.concat([_layer_df(winner['val'], encoder), egemaps['val'], whisper['val']], axis=1)}
                sets[f'F_{encoder}_plus_eGeMAPS_Whisper'] = {'frames': fused, 'kind': 'handcrafted', 'encoder': encoder, 'layer': winner['layer']}
    return sets


def _historical_legacy_rows(targets, config):
    historical = project_path(config['paths'].get('legacy_run_dir', 'outputs/audio/legacy_run')) / 'metrics'
    result = []
    for target in targets:
        path = historical / f'{target}_summary.json'
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding='utf-8'))
        candidate = summary.get('best_single_candidate_by_cv', {})
        val = candidate.get('validation_metrics', {})
        test = summary.get('final_test_metrics', {})
        record = {'target': target, 'feature_set': 'A_legacy_handcrafted (historical)', 'model': candidate.get('model', 'historical'), 'historical_result': True, 'validation_selected': False, 'final_test_metrics': test, 'cv_metrics': summary.get('cv_metrics', {}), 'val_metrics': val, 'feature_columns': candidate.get('features', []), 'params': candidate.get('params', {}), 'bootstrap_ci': None, 'oof_predictions': None}
        result.append(record)
    return result


def reproduce_legacy_cv(data, legacy, config, report_dir, seed, folds, n_jobs=2):
    """Replay each saved legacy winner on train GroupKFold + validation only."""
    legacy_root = project_path(config['paths'].get('legacy_run_dir', 'outputs/audio/legacy_run'))
    rows = []
    if legacy is None:
        return rows
    for target in config['targets'].values():
        summary_path = legacy_root / 'metrics' / f'{target}_summary.json'
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        prior = summary.get('best_single_candidate_by_cv', {})
        model_name = prior.get('model')
        params = prior.get('params', {})
        columns = [name for name in prior.get('features', []) if name in legacy['train'].columns]
        if not model_name or not columns:
            continue
        y_all = pd.to_numeric(data['train'][target], errors='coerce')
        train_mask = y_all.notna().to_numpy()
        y = y_all[train_mask].reset_index(drop=True)
        X = legacy['train'].loc[train_mask, columns].reset_index(drop=True)
        _assert_no_target_columns(X, config, f'legacy reproduction training X for {target}')
        users = data['train'].loc[train_mask, 'user_id'].astype(str).to_numpy()
        _, cv_metrics, fold_results = grouped_oof(X, y, users, model_name, params, folds, seed, standardize=False, n_jobs=n_jobs)
        model = build_pipeline(X, model_name, params, seed, standardize=False, n_jobs=n_jobs)
        model.fit(X, y)
        val_mask = pd.to_numeric(data['val'][target], errors='coerce').notna().to_numpy()
        yv = pd.to_numeric(data['val'].loc[val_mask, target]).reset_index(drop=True)
        Xv = legacy['val'].loc[val_mask, columns].reset_index(drop=True)
        _assert_no_target_columns(Xv, config, f'legacy reproduction validation X for {target}')
        val_metrics = metrics(yv, model.predict(Xv))
        prior_cv = summary.get('cv_metrics', {})
        prior_val = prior.get('validation_metrics', {})
        rows.append({'target': target, 'model': model_name, 'params': json.dumps(params), 'cv_rmse_previous': prior_cv.get('rmse'), 'cv_rmse_reproduced': cv_metrics['rmse'], 'cv_pearson_previous': prior_cv.get('pearson'), 'cv_pearson_reproduced': cv_metrics['pearson'], 'val_rmse_previous': prior_val.get('rmse'), 'val_rmse_reproduced': val_metrics['rmse'], 'val_pearson_previous': prior_val.get('pearson'), 'val_pearson_reproduced': val_metrics['pearson'], 'cv_fold_rmse_mean': float(np.mean([item['rmse'] for item in fold_results])), 'cv_fold_rmse_std': float(np.std([item['rmse'] for item in fold_results])), 'note': 'Replayed on fixed train GroupKFold and validation only; test not read.'})
    if rows:
        atomic_csv(pd.DataFrame(rows), report_dir / 'legacy_reproduction.csv')
    return rows


def _fit_models(data, feature_sets, config, folds, seed, smoke=False, n_jobs=2):
    candidate_records = []
    target_winners = {}
    trial_rows = []
    for target_name, target in config['targets'].items():
        y_all = pd.to_numeric(data['train'][target], errors='coerce')
        train_mask = y_all.notna().to_numpy()
        y = y_all[train_mask].reset_index(drop=True)
        groups = data['train'].loc[train_mask, 'user_id'].astype(str).to_numpy()
        target_candidates = []
        for set_name, definition in feature_sets.items():
            train_frame = definition['frames']['train'].loc[train_mask].reset_index(drop=True)
            val_all = definition['frames']['val']
            _assert_no_target_columns(train_frame, config, f'{set_name}/{target_name} training X')
            _assert_no_target_columns(val_all, config, f'{set_name}/{target_name} validation X')
            constants = [c for c in train_frame if c != 'question_id' and train_frame[c].nunique(dropna=True) <= 1]
            train_frame = train_frame.drop(columns=constants, errors='ignore')
            val_all = val_all.drop(columns=constants, errors='ignore')
            seen_hashes = {}
            duplicates = []
            for column in [c for c in train_frame if c != 'question_id']:
                token = hashlib.blake2b(pd.util.hash_pandas_object(train_frame[column], index=False).values.tobytes(), digest_size=16).digest()
                previous = next((name for name in seen_hashes.get(token, []) if train_frame[column].equals(train_frame[name])), None)
                if previous is not None:
                    duplicates.append(column)
                else:
                    seen_hashes.setdefault(token, []).append(column)
            train_frame = train_frame.drop(columns=duplicates, errors='ignore')
            val_all = val_all.drop(columns=duplicates, errors='ignore')
            val_y_all = pd.to_numeric(data['val'][target], errors='coerce')
            val_mask = val_y_all.notna().to_numpy()
            val_y = val_y_all[val_mask].reset_index(drop=True)
            X_val = val_all.loc[val_mask].reset_index(drop=True)
            feature_columns = [c for c in train_frame.columns if c != 'question_id']
            if not feature_columns or train_frame[feature_columns].isna().all().all():
                print(f'[{set_name}][{target_name}] SKIPPED: no finite training features', flush=True)
                continue
            question_variants = [False, True] if config['modeling'].get('question_id_ablation', False) else [False]
            kind = definition['kind']
            stable_ridge = bool(definition.get('encoder')) or kind == 'embedding'
            branch_name = definition.get('encoder') or {
                'A_legacy_handcrafted': 'legacy', 'B_eGeMAPS': 'egemaps',
                'C_eGeMAPS_plus_Whisper': 'egemaps+whisper'
            }.get(set_name, set_name.split('_', 1)[0].lower())
            for use_question in question_variants:
                matrix = train_frame.copy()
                val_matrix = X_val.copy()
                if use_question:
                    matrix['question_id'] = data['train'].loc[train_mask, 'question_id'].astype(str).to_numpy()
                    val_matrix['question_id'] = data['val'].loc[val_mask, 'question_id'].astype(str).to_numpy()
                else:
                    matrix.drop(columns=['question_id'], errors='ignore', inplace=True)
                    val_matrix.drop(columns=['question_id'], errors='ignore', inplace=True)
                _assert_no_target_columns(matrix, config, f'{set_name}/{target_name} model training X')
                _assert_no_target_columns(val_matrix, config, f'{set_name}/{target_name} model validation X')
                for model_name in available_models(kind, stable_ridge):
                    try:
                        best, oof, trials = search_model(
                            matrix, y, groups, model_name, config, folds, seed,
                            standardize=model_name in ('Ridge', 'SVR') or kind == 'embedding',
                            question_id=use_question,
                            pca_components=config['modeling'].get('embedding_pca_components') if kind == 'embedding' else None,
                            n_jobs=n_jobs, stable_ridge=stable_ridge)
                        for trial in trials:
                            trial_rows.append({'target': target, 'feature_set': set_name,
                                               'question_id_ablation': use_question, **trial})
                        if best is None:
                            errors = [trial.get('error', '') for trial in trials if trial.get('error')]
                            reason = errors[-1] if errors else 'all CV trials failed'
                            print(f'[{branch_name}][{target_name}] {model_name} FAILED: {reason}', flush=True)
                            continue
                        val_pipeline = build_pipeline(
                            matrix, model_name, best['params'], seed,
                            standardize=model_name in ('Ridge', 'SVR') or kind == 'embedding',
                            question_id=use_question,
                            pca_components=config['modeling'].get('embedding_pca_components') if kind == 'embedding' else None,
                            n_jobs=n_jobs, stable_ridge=stable_ridge)
                        val_pipeline.fit(matrix, y)
                        val_pred = val_pipeline.predict(val_matrix)
                        val_scores = metrics(val_y, val_pred)
                    except Exception as exc:
                        message = f'{type(exc).__name__}: {exc}'
                        trial_rows.append({'target': target, 'feature_set': set_name,
                                           'question_id_ablation': use_question, 'model': model_name,
                                           'error': message})
                        print(f'[{branch_name}][{target_name}] {model_name} FAILED: {message}', flush=True)
                        continue
                    oof_scores = {k: best[f'cv_{k}'] for k in ('mae', 'rmse', 'r2', 'pearson', 'spearman', 'ccc') if f'cv_{k}' in best}
                    final_set_name = set_name + ('+question_id' if use_question else '')
                    candidate = _candidate_record(target, final_set_name, model_name, best, val_scores, oof, oof_scores, matrix.columns, use_question)
                    candidate.update({'pipeline': val_pipeline, 'kind': kind, 'constant_features_removed': constants, 'duplicate_features_removed': duplicates, 'pca_components': config['modeling'].get('embedding_pca_components') if kind == 'embedding' else None, 'encoder': definition.get('encoder'), 'layer': definition.get('layer')})
                    candidate_records.append(candidate)
                    target_candidates.append(candidate)
                    print(f'[{branch_name}][{target_name}] {model_name} complete: '
                          f'val RMSE={val_scores["rmse"]:.4f}, val MAE={val_scores["mae"]:.4f}, '
                          f'val R2={val_scores["r2"]:.4f}', flush=True)
        if target_candidates:
            target_winners[target] = max(target_candidates, key=lambda row: (row['val_metrics'].get('ccc', -math.inf), -row['val_metrics'].get('rmse', math.inf)))
            target_winners[target]['validation_selected'] = True
    return (candidate_records, target_winners, trial_rows)


def _final_test(data, winners, feature_sets, config, seed, folds, smoke, bootstrap_replicates, n_jobs):
    if smoke:
        LOG.info('Smoke mode: final test evaluation intentionally skipped')
        return
    for target, selected in winners.items():
        set_name = selected['feature_set'].removesuffix('+question_id')
        definition = feature_sets[target][set_name]
        train, val, test = (definition['frames'][s] for s in ('train', 'val', 'test'))
        if selected['question_id_ablation']:
            train = train.copy()
            val = val.copy()
            test = test.copy()
            train['question_id'] = data['train'].question_id.astype(str).to_numpy()
            val['question_id'] = data['val'].question_id.astype(str).to_numpy()
            test['question_id'] = data['test'].question_id.astype(str).to_numpy()
        selected_columns = list(selected['feature_columns'])
        train, val, test = (frame[selected_columns].copy() for frame in (train, val, test))
        _assert_no_target_columns(train, config, f'{set_name}/{target} final training X')
        _assert_no_target_columns(val, config, f'{set_name}/{target} final validation X')
        _assert_no_target_columns(test, config, f'{set_name}/{target} final test X')
        train_ids = pd.to_numeric(data['train'][target], errors='coerce').notna().to_numpy()
        val_ids = pd.to_numeric(data['val'][target], errors='coerce').notna().to_numpy()
        test_y_all = pd.to_numeric(data['test'][target], errors='coerce')
        test_mask = test_y_all.notna().to_numpy()
        X_trainval = pd.concat([train.loc[train_ids], val.loc[val_ids]], ignore_index=True)
        y_trainval = pd.concat([pd.to_numeric(data['train'].loc[train_ids, target]), pd.to_numeric(data['val'].loc[val_ids, target])], ignore_index=True)
        X_test = test.loc[test_mask].reset_index(drop=True)
        y_test = test_y_all[test_mask].reset_index(drop=True)
        if selected['question_id_ablation']:
            X_trainval['question_id'] = pd.concat([data['train'].loc[train_ids, 'question_id'].astype(str), data['val'].loc[val_ids, 'question_id'].astype(str)], ignore_index=True)
            X_test['question_id'] = data['test'].loc[test_mask, 'question_id'].astype(str).to_numpy()
        else:
            X_trainval = X_trainval.drop(columns=['question_id'], errors='ignore')
            X_test = X_test.drop(columns=['question_id'], errors='ignore')
        try:
            model = build_pipeline(X_trainval, selected['model'], selected['cv_record']['params'], seed, standardize=selected['model'] in ('Ridge', 'SVR') or selected['kind'] == 'embedding', question_id=selected['question_id_ablation'], pca_components=selected['pca_components'], n_jobs=n_jobs, stable_ridge=bool(selected.get('encoder')) or selected['kind'] == 'embedding')
            model.fit(X_trainval, y_trainval)
            test_pred = model.predict(X_test)
        except Exception as exc:
            message = f'{type(exc).__name__}: {exc}'
            selected['final_test_error'] = message
            print(f"[{selected.get('encoder') or set_name}][{target}] final test FAILED: {message}", flush=True)
            continue
        user_test = data['test'].loc[test_mask, 'user_id'].astype(str).to_numpy()
        selected['final_test_metrics'] = metrics(y_test, test_pred)
        selected['bootstrap_ci'] = bootstrap_by_participant(y_test, test_pred, user_test, int(bootstrap_replicates), int(config['modeling'].get('bootstrap_seed', seed + 1)))
        selected['test_predictions'] = pd.DataFrame({'id': data['test'].loc[test_mask, 'id'].to_numpy(), 'user_id': user_test, 'question_id': data['test'].loc[test_mask, 'question_id'].to_numpy(), 'split': 'test', 'target': target, 'actual': y_test, 'prediction': test_pred})
        selected['final_model'] = model
        selected['final_training_rows'] = int(len(y_trainval))


def run_comparison(config, limit=None, seed=None, n_jobs=2, trials=None):
    seed = int(seed if seed is not None else config['seed'])
    smoke = limit is not None
    report_dir = project_path(config['paths']['reports_dir']) / ('smoke' if smoke else '')
    models_dir = project_path(config['paths']['models_dir']) / ('smoke' if smoke else '')
    report_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    data = load_split_data(config, limit=limit)
    diagnostics = label_diagnostics(data, config)
    atomic_json(report_dir / 'label_diagnostics.json', diagnostics)
    folds = int(config['modeling'].get('smoke_cv_folds', 3) if smoke else config['modeling'].get('cv_folds', 5))
    if data['train'].user_id.nunique() < folds:
        raise ValueError(f"Need at least {folds} training participants for grouped CV; found {data['train'].user_id.nunique()}")
    legacy = load_feature_group('legacy', data, config, smoke)
    egemaps = load_feature_group('egemaps', data, config, smoke)
    whisper = load_feature_group('whisper', data, config, smoke)
    encoders = {name: load_embeddings(name, data, config, smoke, splits=('train', 'val')) for name in ('wavlm', 'hubert')}
    if legacy is None and egemaps is None and (whisper is None) and (not any(encoders.values())):
        raise FileNotFoundError('No usable feature tables/embeddings were found; run scripts/extract_audio_features.py first')
    if not smoke:
        reproduce_legacy_cv(data, legacy, config, report_dir, seed, folds, n_jobs)
    candidate_sets_by_target = {}
    all_records, winners, all_trials = ([], {}, [])
    for target_name, target in config['targets'].items():
        encoder_winners = {}
        for encoder in ('wavlm', 'hubert'):
            if encoders[encoder] is not None:
                try:
                    encoder_winners[encoder] = _select_encoder_layer(data, encoders[encoder], encoder, config, folds, seed, report_dir, target)
                except Exception as exc:
                    print(f'[{encoder}][{target_name}] layer sweep FAILED: {type(exc).__name__}: {exc}', flush=True)
        feature_sets = _feature_sets_for_target(data, legacy, egemaps, whisper, encoder_winners)
        for definition in feature_sets.values():
            definition['test'] = None
        records, target_winner, trials = _fit_models_for_target(data, feature_sets, config, target, folds, seed, smoke, n_jobs, trials)
        all_records.extend(records)
        all_trials.extend(trials)
        if target in target_winner:
            target_winner[target]['encoder_winners'] = encoder_winners
            winners[target] = target_winner[target]
            candidate_sets_by_target[target] = feature_sets
    for target, selected in winners.items():
        definition = candidate_sets_by_target[target][selected['feature_set'].removesuffix('+question_id')]
        selected['feature_sets_ref'] = definition
        if definition['kind'] == 'embedding' or definition.get('encoder'):
            encoder = definition['encoder']
            full = load_embeddings(encoder, data, config, smoke, splits=('test',))
            pooled_test = full['test']
            index_path = project_path(config['paths']['embeddings_dir']) / ('smoke' if smoke else '') / encoder / 'test_embeddings.npz'
            with np.load(index_path, allow_pickle=False) as saved:
                saved_ids = saved['ids'].astype(str).tolist()
                saved_pooled = saved['pooled'].astype(np.float32)
            layer_label = definition['layer']
            if layer_label.startswith('mean_last_'):
                n = int(layer_label.rsplit('_', 1)[-1])
                selected_pooled = pooled_test[:, -n:, :].mean(axis=1)
            else:
                selected_pooled = pooled_test[:, int(layer_label.removeprefix('layer_')), :]
            selected_test = _layer_df(selected_pooled, encoder)
            if definition['frames'].get('train') is not None and len(definition['frames']['train'].columns) > selected_test.shape[1]:
                extra_names = [c for c in definition['frames']['train'].columns if c not in selected_test.columns and c != 'question_id']
                extra_frames = []
                for extractor in ('egemaps', 'whisper'):
                    branch = load_feature_group(extractor, data, config, smoke)
                    if branch is not None:
                        extra_frames.append(branch['test'])
                if extra_frames:
                    selected_test = pd.concat([selected_test, *extra_frames], axis=1)
            definition['frames']['test'] = selected_test
        else:
            base_name = selected['feature_set'].removesuffix('+question_id')
            if base_name == 'A_legacy_handcrafted':
                definition['frames']['test'] = legacy['test']
            elif base_name == 'B_eGeMAPS':
                definition['frames']['test'] = egemaps['test']
            elif base_name == 'C_eGeMAPS_plus_Whisper':
                definition['frames']['test'] = pd.concat([egemaps['test'], whisper['test']], axis=1)
            elif base_name.startswith('F_'):
                definition['frames']['test'] = pd.concat([egemaps['test'], whisper['test']], axis=1)
            else:
                definition['frames']['test'] = legacy['test']
    selected_map = {target: row for target, row in winners.items()}
    _final_test(data, selected_map, candidate_sets_by_target, config, seed, folds, smoke, config['modeling']['bootstrap_replicates'], n_jobs)
    for target, selected in winners.items():
        base_name = selected['feature_set'].removesuffix('+question_id')
        pred_dir = report_dir / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        train_oof = selected['oof_predictions']
        train_rows = data['train'].loc[pd.to_numeric(data['train'][target], errors='coerce').notna()]
        atomic_csv(pd.DataFrame({'id': train_rows.id.to_numpy(), 'user_id': train_rows.user_id.to_numpy(), 'question_id': train_rows.question_id.to_numpy(), 'split': 'train', 'target': target, 'actual': pd.to_numeric(train_rows[target]).to_numpy(), 'oof_prediction': train_oof}), pred_dir / f'{target}_oof_predictions.csv')
        valframe = candidate_sets_by_target[target][base_name]['frames']['val']
        val_rows = data['val'].loc[pd.to_numeric(data['val'][target], errors='coerce').notna()]
        Xv = valframe.loc[pd.to_numeric(data['val'][target], errors='coerce').notna()].reset_index(drop=True)
        if not selected['question_id_ablation']:
            Xv = Xv.drop(columns=['question_id'], errors='ignore')
        val_prediction = selected['pipeline'].predict(Xv)
        atomic_csv(pd.DataFrame({'id': val_rows.id.to_numpy(), 'user_id': val_rows.user_id.to_numpy(), 'question_id': val_rows.question_id.to_numpy(), 'split': 'val', 'target': target, 'actual': pd.to_numeric(val_rows[target]).to_numpy(), 'prediction': val_prediction}), pred_dir / f'{target}_val_predictions.csv')
        if selected.get('test_predictions') is not None:
            atomic_csv(selected['test_predictions'], pred_dir / f'{target}_test_predictions.csv')
        model_path = models_dir / target / f"{base_name}_{selected['model']}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import joblib
            joblib.dump(selected.get('final_model', selected['pipeline']), model_path)
        except ImportError:
            warnings.warn('joblib not installed; fitted models not serialized')
        sidecar = {'schema_version': 1, 'clip_key': 'id', 'target': target, 'feature_set': base_name, 'model': selected['model'], 'params': selected['cv_record']['params'], 'feature_columns': selected['feature_columns'], 'question_id_ablation': selected['question_id_ablation'], 'extractors': ['wavlm' if 'WavLM' in base_name else 'hubert', 'egemaps', 'whisper'] if base_name.startswith('F_') else ['wavlm' if base_name.startswith('D_') else 'hubert'] if base_name.startswith(('D_', 'E_')) else ['egemaps', 'whisper'] if base_name.startswith('C_') else ['egemaps'] if base_name.startswith('B_') else ['legacy'], 'encoder': selected.get('encoder'), 'layer': selected.get('layer'), 'pooling': 'mean+std over transformer frames' if selected.get('encoder') else None, 'training_participants': int(data['train'].user_id.nunique()), 'scaler_fitted_inside_pipeline': True, 'fit_splits': ['train', 'val'] if not smoke else ['train'], 'model_path': str(model_path.relative_to(ROOT).as_posix())}
        atomic_json(models_dir / target / 'fusion_schema.json', sidecar)
    output_rows = []
    for record in all_records:
        output_rows.append({'target': record['target'], 'feature_set': record['feature_set'], 'model': record['model'], 'question_id_ablation': record['question_id_ablation'], **{f'cv_{k}': v for k, v in record['cv_metrics'].items()}, **{f'val_{k}': v for k, v in record['val_metrics'].items()}, 'validation_selected': record['validation_selected'], **{f'test_{k}': v for k, v in (record['final_test_metrics'] or {}).items()}, 'params': json.dumps(record['cv_record']['params']), 'features': json.dumps(record['feature_columns']), 'historical_result': False})
    for record in _historical_legacy_rows(config['targets'].values(), config):
        output_rows.append({'target': record['target'], 'feature_set': record['feature_set'], 'model': record['model'], **{f'cv_{k}': v for k, v in record['cv_metrics'].items()}, **{f'val_{k}': v for k, v in record['val_metrics'].items()}, **{f'historical_test_{k}': v for k, v in record['final_test_metrics'].items()}, 'historical_result': True, 'params': json.dumps(record['params']), 'features': json.dumps(record['feature_columns'])})
    comparison = pd.DataFrame(output_rows)
    atomic_csv(comparison, report_dir / 'comparison.csv')
    try:
        markdown = comparison.to_markdown(index=False)
    except ImportError:
        markdown = '\n'.join(['| ' + ' | '.join(map(str, comparison.columns)) + ' |', '| ' + ' | '.join(['---'] * len(comparison.columns)) + ' |'] + ['| ' + ' | '.join((str(value) for value in row)) + ' |' for row in comparison.itertuples(index=False, name=None)])
    (report_dir / 'comparison.md').write_text(markdown, encoding='utf-8')
    if all_trials:
        atomic_csv(pd.DataFrame(all_trials), report_dir / 'cv_trials.csv')
    summary = {'seed': seed, 'cv_strategy': 'GroupKFold by user_id', 'cv_folds': folds, 'test_policy': 'One final evaluation per validation-selected target configuration; smoke mode skips test', 'selected': {target: {'feature_set': row['feature_set'], 'model': row['model'], 'params': row['cv_record']['params'], 'cv_metrics': row['cv_metrics'], 'validation_metrics': row['val_metrics'], 'final_test_metrics': row['final_test_metrics'], 'bootstrap_ci': row['bootstrap_ci']} for target, row in winners.items()}, 'skipped_feature_sets': [name for name, result in (('legacy', legacy), ('egemaps', egemaps), ('whisper', whisper), ('wavlm', encoders['wavlm']), ('hubert', encoders['hubert'])) if result is None], 'smoke_limit_per_split': limit}
    atomic_json(report_dir / 'comparison_summary.json', summary)
    _plots(comparison, winners, report_dir)
    return {'comparison': comparison, 'winners': winners, 'data': data, 'feature_sets': candidate_sets_by_target}


def _fit_models_for_target(data, feature_sets, config, target, folds, seed, smoke, n_jobs, trial_override):
    one = dict(config)
    one['targets'] = {name: col for name, col in config['targets'].items() if col == target}
    return _fit_models(data, feature_sets, one, folds, seed, smoke, n_jobs)


def print_comparison_table(comparison):
    """Always print compact validation metrics, including an empty header on failure."""
    columns = ['feature_set', 'model', 'target', 'val_mae', 'val_rmse', 'val_r2', 'val_pearson', 'val_spearman']
    headers = ['Feature set', 'Model', 'Target', 'MAE', 'RMSE', 'R2', 'Pearson', 'Spearman']
    if comparison is not None and not comparison.empty:
        rows = comparison.copy()
        if 'historical_result' in rows:
            rows = rows[~rows['historical_result'].fillna(False)]
        table = rows[columns].rename(columns=dict(zip(columns, headers))) if all(name in rows for name in columns) else pd.DataFrame(columns=headers)
    else:
        table = pd.DataFrame(columns=headers)
    print('\nVALIDATION COMPARISON', flush=True)
    if table.empty:
        print('(No models completed successfully.)', flush=True)
        print(' | '.join(headers), flush=True)
    else:
        print(table.to_string(index=False, float_format=lambda value: f'{value:.4f}'), flush=True)


def _plots(comparison, winners, report_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        (report_dir / 'PLOTS_UNAVAILABLE.txt').write_text('Install matplotlib to create metric, layer-sweep and prediction plots.\n', encoding='utf-8')
        return
    plot_dir = report_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    historical_mask = comparison['historical_result'].fillna(False) if 'historical_result' in comparison else pd.Series(False, index=comparison.index)
    selected_val = comparison[~historical_mask]
    for metric_name in ('ccc', 'rmse'):
        column = f'val_{metric_name}'
        if column not in selected_val:
            continue
        for target, group in selected_val.groupby('target'):
            summary = group.groupby('feature_set')[column].max() if metric_name == 'ccc' else group.groupby('feature_set')[column].min()
            fig, ax = plt.subplots(figsize=(10, 5))
            summary.plot(kind='bar', ax=ax)
            ax.set_ylabel(f'Validation {metric_name.upper()}')
            ax.set_title(f'Validation comparison: {target}')
            fig.tight_layout()
            fig.savefig(plot_dir / f'{target}_validation_{metric_name}.png', dpi=160)
            plt.close(fig)
    for target, record in winners.items():
        pred = record.get('test_predictions')
        if pred is None:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pred.actual, pred.prediction, alpha=0.65, s=18)
        bounds = [min(pred.actual.min(), pred.prediction.min()), max(pred.actual.max(), pred.prediction.max())]
        ax.plot(bounds, bounds, '--', color='gray')
        ax.set(xlabel='Actual', ylabel='Predicted', title=f'Final test predictions: {target}')
        fig.tight_layout()
        fig.savefig(plot_dir / f'{target}_test_scatter.png', dpi=160)
        plt.close(fig)
        test_column = 'test_ccc'
        if test_column in comparison:
            tested = comparison[(comparison.target == target) & comparison[test_column].notna() & ~comparison.get('historical_result', pd.Series(False, index=comparison.index)).fillna(False)]
            if not tested.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(tested.feature_set, tested[test_column], color='#a7683c')
                ax.set_ylabel('Final test CCC')
                ax.set_title(f'Final test configuration only: {target}')
                ax.tick_params(axis='x', rotation=25)
                fig.tight_layout()
                fig.savefig(plot_dir / f'{target}_final_test_ccc.png', dpi=160)
                plt.close(fig)
        for encoder in ('wavlm', 'hubert'):
            curve_path = report_dir / f'layer_sweep_{encoder}_{target}.csv'
            if curve_path.is_file():
                curve = pd.read_csv(curve_path)
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(curve.layer.astype(str), curve.cv_rmse_fold_mean, marker='o')
                ax.set(xlabel='Layer / last-N average', ylabel='Grouped CV RMSE', title=f'{encoder} layer sweep: {target}')
                ax.tick_params(axis='x', rotation=60)
                fig.tight_layout()
                fig.savefig(plot_dir / f'{encoder}_{target}_layer_sweep.png', dpi=160)
                plt.close(fig)


def compare_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=None)
    parser.add_argument('--limit', type=int, default=None, help='Smoke mode: first N clips per split; skips test metrics')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=2)
    parser.add_argument('--trials', type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.trials is not None:
        config['modeling']['tree_trials'] = args.trials
    run_comparison(config, args.limit, args.seed, args.n_jobs, args.trials)
    return 0


'Fusion-ready train OOF/validation/test representations and single-WAV API.'


def _feature_spec(feature_set, encoder):
    if feature_set.startswith('A_'):
        return ['legacy']
    if feature_set.startswith('B_'):
        return ['egemaps']
    if feature_set.startswith('C_'):
        return ['egemaps', 'whisper']
    if feature_set.startswith('D_'):
        return [encoder or 'wavlm']
    if feature_set.startswith('E_'):
        return [encoder or 'hubert']
    if feature_set.startswith('F_'):
        return [encoder or 'wavlm', 'egemaps', 'whisper']
    raise ValueError(f'Unrecognized selected audio feature set: {feature_set}')


def _fit_training_transform(train_frame):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    numeric_columns = [name for name in train_frame.columns if name != 'question_id']
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    train_imputed = imputer.fit_transform(train_frame[numeric_columns])
    scaler = StandardScaler().fit(train_imputed)
    indicator_indices = imputer.indicator_.features_.astype(int).tolist()
    names = numeric_columns + [f'{numeric_columns[i]}__missing' for i in indicator_indices]
    return (numeric_columns, imputer, scaler, names, indicator_indices)


def export_fusion(config, comparison_result=None, limit=None):
    if comparison_result is None:
        raise RuntimeError('Fusion export needs the in-memory, validation-selected comparison result. Run scripts/compare_audio_models.py.')
    smoke = limit is not None
    output = project_path(config['paths'].get('fusion_dir', 'outputs/audio/fusion')) / ('smoke' if smoke else '')
    output.mkdir(parents=True, exist_ok=True)
    winners = comparison_result['winners']
    feature_sets = comparison_result['feature_sets']
    data = comparison_result['data']
    report_root = project_path(config['paths']['reports_dir']) / ('smoke' if smoke else '')
    for target, selected in winners.items():
        feature_set = selected['feature_set'].removesuffix('+question_id')
        definition = feature_sets[target][feature_set]
        frames = definition['frames']
        selected_columns = list(selected['feature_columns'])
        train_frame = frames['train'].reset_index(drop=True).copy()
        if selected['question_id_ablation']:
            train_frame['question_id'] = data['train'].question_id.astype(str).to_numpy()
        train_frame = train_frame[selected_columns]
        feature_names, imputer, scaler, output_names, indicator_indices = _fit_training_transform(train_frame)
        question_categories = sorted(data['train'].question_id.astype(str).unique().tolist()) if selected['question_id_ablation'] else []
        full_output_names = output_names + [f'question_id={value}' for value in question_categories]
        target_dir = output / target
        target_dir.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'val', 'test'):
            raw_frame = frames[split].reset_index(drop=True)
            if selected['question_id_ablation']:
                raw_frame['question_id'] = data[split].question_id.astype(str).to_numpy()
            raw_frame = raw_frame[selected_columns]
            values = raw_frame[feature_names].to_numpy(dtype=float)
            model_input = scaler.transform(imputer.transform(raw_frame[feature_names]))
            meta = data[split].reset_index(drop=True)
            if question_categories:
                q_values = meta.question_id.astype(str).to_numpy()
                q_onehot = np.asarray([[float(value == category) for category in question_categories] for value in q_values], dtype=np.float32)
                model_input = np.column_stack([model_input, q_onehot])
            atomic_npz(target_dir / f'{split}_audio_features.npz', ids=meta.id.astype(str).to_numpy(), user_ids=meta.user_id.astype(str).to_numpy(), question_ids=meta.question_id.astype(str).to_numpy(), split=np.asarray(split), feature_names=np.asarray(feature_names), raw_features=values.astype(np.float32), model_input=model_input.astype(np.float32), model_input_names=np.asarray(full_output_names))
        sidecar = {'schema_version': 1, 'join_key': 'id', 'participant_key': 'user_id', 'question_key': 'question_id', 'feature_set': feature_set, 'extractors': _feature_spec(feature_set, selected.get('encoder')), 'model': selected['model'], 'target': target, 'checkpoint': config['extractors'].get(selected.get('encoder'), {}).get('model_name') if selected.get('encoder') else None, 'selected_layer': selected.get('layer'), 'pooling': 'time mean concatenated with standard deviation', 'feature_names': feature_names, 'model_input_feature_names': full_output_names, 'missing_indicator_indices': indicator_indices, 'question_categories_train_only': question_categories, 'question_id_ablation': selected['question_id_ablation'], 'train_only_imputer_median': imputer.statistics_.tolist(), 'train_only_scaler_mean': scaler.mean_.tolist(), 'train_only_scaler_scale': scaler.scale_.tolist(), 'scaler_fit_split': 'train only', 'split_files': {split: f'{split}_audio_features.npz' for split in ('train', 'val', 'test')}, 'prediction_files': {'train': f'{target}_oof_predictions.csv', 'val': f'{target}_val_predictions.csv', 'test': f'{target}_test_predictions.csv'}, 'split_policy': 'RecruitView fixed participant-level splits; train OOF predictions use GroupKFold by user_id'}
        atomic_json(target_dir / 'schema.json', sidecar)
        prediction_dir = target_dir / 'predictions'
        prediction_dir.mkdir(exist_ok=True)
        for split, filename in (('train', f'{target}_oof_predictions.csv'), ('val', f'{target}_val_predictions.csv'), ('test', f'{target}_test_predictions.csv')):
            source = report_root / 'predictions' / filename
            if source.is_file():
                prediction_dir.joinpath(filename).write_bytes(source.read_bytes())
    atomic_json(output / 'index.json', {'export_schema_version': 1, 'targets': list(winners), 'source': 'validation-selected audio-only representations', 'smoke_limit_per_split': limit, 'join_key': 'id', 'participant_key': 'user_id'})
    return output


def single_wav_to_representation(wav_path, sidecar_path, config=None, allow_download=False, question_id: str | None=None):
    """Extract and transform one deployment WAV using its selected feature schema."""
    config = config or load_config()
    sidecar_path = Path(sidecar_path)
    if not sidecar_path.is_absolute():
        sidecar_path = ROOT / sidecar_path
    sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
    if sidecar.get('question_id_ablation') and question_id is None:
        raise ValueError('This schema uses question_id; pass question_id to compute the deployment representation')
    path = Path(wav_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    signal, sr, duration = _audio(path)
    values = {}
    for extractor_name in sidecar['extractors']:
        if extractor_name == 'legacy':
            values.update(LegacyWaveformExtractor().extract(path))
        elif extractor_name == 'egemaps':
            values.update(EGeMAPSExtractor(config).extract(path))
        elif extractor_name == 'whisper':
            model = WhisperTimingExtractor(config, project_path(config['paths']['cache_dir']) / 'whisper', allow_download)
            (result, evidence), _raw = model.extract(path, duration, project_path(config['paths']['cache_dir']) / 'single_wav_whisper.json', use_cache=False)
            values.update(result)
        elif extractor_name in ('wavlm', 'hubert'):
            model = TransformerEmbeddingExtractor(extractor_name, config, project_path(config['paths']['cache_dir']) / 'huggingface', allow_download)
            temp = project_path(config['paths']['cache_dir']) / 'single_wav_embedding.npz'
            pooled = model.extract(signal, sr, temp, use_cache=False)
            layer = sidecar.get('selected_layer') or 'best_cv'
            if str(layer).startswith('mean_last_'):
                n = int(str(layer).rsplit('_', 1)[-1])
                vector = pooled[-n:].mean(axis=0)
            else:
                vector = pooled[int(str(layer).removeprefix('layer_'))]
            values.update({f'{extractor_name}_{i}': float(value) for i, value in enumerate(vector)})
    names = sidecar['feature_names']
    raw = np.asarray([values.get(name, np.nan) for name in names], dtype=float).reshape(1, -1)
    medians = np.asarray(sidecar['train_only_imputer_median'], dtype=float)
    missing = ~np.isfinite(raw)
    raw = np.where(missing, medians[None, :], raw)
    indicator_indices = np.asarray(sidecar.get('missing_indicator_indices', []), dtype=int)
    if indicator_indices.size:
        raw = np.concatenate([raw, missing[:, indicator_indices].astype(float)], axis=1)
    mean = np.asarray(sidecar['train_only_scaler_mean'], dtype=float)
    scale = np.asarray(sidecar['train_only_scaler_scale'], dtype=float)
    transformed = (raw - mean) / np.where(scale == 0, 1.0, scale)
    categories = sidecar.get('question_categories_train_only', [])
    if categories:
        q_onehot = np.asarray([[float(str(question_id) == category) for category in categories]], dtype=float)
        transformed = np.column_stack([transformed, q_onehot])
    return (transformed[0].astype(np.float32), sidecar['model_input_feature_names'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare audio feature branches and export the selected fusion vectors.')
    parser.add_argument('--config', default=None)
    parser.add_argument('--limit', type=int, default=None, help='Smoke mode: first N clips per split; skips test scoring')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--n-jobs', type=int, default=2)
    parser.add_argument('--trials', type=int, default=None)
    args = parser.parse_args(argv)
    try:
        import torch
        if torch.cuda.is_available():
            print(f'[gpu] CUDA available: {torch.cuda.get_device_name(0)}. Cached embeddings will be reused; '
                  'sklearn Ridge runs on CPU, and XGBoost uses CUDA when a compatible version is installed.', flush=True)
    except ImportError:
        pass
    result = None
    try:
        config = load_config(args.config)
        if args.trials is not None:
            config['modeling']['tree_trials'] = args.trials
        result = run_comparison(config, args.limit, args.seed, args.n_jobs, args.trials)
        export_fusion(config, result, args.limit)
        print_comparison_table(result['comparison'])
        return 0
    except Exception as exc:
        print(f'COMPARISON FAILED: {type(exc).__name__}: {exc}', flush=True)
        print_comparison_table(result['comparison'] if result is not None else pd.DataFrame())
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
