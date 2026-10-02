from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import logging
import math
import numpy as np
import os
import pandas as pd
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
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
    excluded = (set(ID_COLUMNS) | {'user_no', 'file_name', 'original_video_path', 'audio_path', 'status',
                'feature_status', 'feature_error', *config['targets'].keys(), *config['targets'].values()})
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


def main(argv=None):
    return extract_main(argv)

if __name__ == '__main__':
    raise SystemExit(main())
