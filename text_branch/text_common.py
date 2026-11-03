"""Shared helpers for the RecruitView text branch: data loading, transcript cleaning,
interpretable linguistic features, and regression metrics.

Inputs are the question text and the Whisper transcript only. `gemini_summary` is never
used as a model input: it was written by an LLM that watched the video and discusses
traits, so it would leak annotation-like information into the text branch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = ROOT / "split_dataset"          # shared participant-level split (source of truth)
OUT_DIR = ROOT / "outputs" / "text"
SPLITS = ("train", "val", "test")
SEED = 42

# RecruitView labels predicted by the text branch (original column names preserved):
# answer_score is text's own target; the other three are the audio branch's targets, so
# fusion can later compare branches on the same labels. facial_expression (video's job)
# and interview_score (optional, overlaps overall_performance) are deliberately excluded.
TARGETS = [
    "answer_score",          # primary text target
    "confidence_score",
    "speaking_skills",
    "overall_performance",
]
ID_COLS = ["id", "user_no", "question_id"]

_TS = re.compile(r"^\[(\d+):(\d+) - (\d+):(\d+)\]\s*")


# --------------------------------------------------------------------------- loading

def load_splits(split_dir: Path = SPLIT_DIR) -> dict[str, pd.DataFrame]:
    """Load the fixed train/val/test CSVs and verify there is no participant leakage."""
    dtype = {"id": str, "user_no": str, "question_id": str}
    dfs = {s: pd.read_csv(split_dir / f"{s}.csv", dtype=dtype, keep_default_na=False,
                          na_values={t: [""] for t in TARGETS}) for s in SPLITS}
    users = {s: set(d["user_no"]) for s, d in dfs.items()}
    assert not users["train"] & users["val"], "participant leakage train/val"
    assert not users["train"] & users["test"], "participant leakage train/test"
    assert not users["val"] & users["test"], "participant leakage val/test"
    all_ids = pd.concat([d["id"] for d in dfs.values()])
    assert all_ids.is_unique, "duplicate response ids across splits"
    for s, d in dfs.items():
        d["split"] = s
        d["answer_text"] = d["transcript"].map(clean_transcript)
    return dfs


def parse_segments(transcript: str) -> list[tuple[float, float, str]]:
    """Return (start_s, end_s, text) for each timestamped transcript line."""
    segs = []
    for line in (transcript or "").splitlines():
        m = _TS.match(line.strip())
        if not m:
            continue
        a, b, c, d = map(int, m.groups())
        segs.append((a * 60 + b, c * 60 + d, line.strip()[m.end():].strip()))
    return segs


def clean_transcript(transcript: str) -> str:
    """Strip timestamps and join segments into one answer string."""
    segs = parse_segments(transcript)
    if segs:
        return " ".join(t for _, _, t in segs if t).strip()
    return (transcript or "").strip()


# --------------------------------------------------------------------------- features
# Lexicons are deliberately small and descriptive. They count observable wording, not
# internal states. Delivery features (speech rate, pauses, fillers, repetitions) are left
# to the audio branch, which measures them from word-level timestamps; Whisper-large-v3
# transcripts also suppress most "um/uh", so they would undercount disfluency anyway.

HEDGES = ["i think", "i guess", "i believe", "i feel", "maybe", "perhaps", "probably",
          "possibly", "kind of", "sort of", "somewhat", "might", "not sure", "i suppose"]
CONNECTIVES = ["first", "firstly", "second", "secondly", "then", "next", "finally", "lastly",
               "because", "therefore", "so that", "as a result", "for example", "for instance",
               "however", "although", "in the end", "eventually", "which led", "this helped"]
# "first"/"second" are counted as connectives only. "one" is matched separately so that
# "no one" and "one of" are not counted as numbers.
NUMBER_WORDS = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "hundred", "thousand", "percent", "third"]
_ONE = re.compile(r"(?<!\bno )\bone\b(?! of\b)")
FP_SING = {"i", "me", "my", "mine", "myself"}
FP_PLUR = {"we", "us", "our", "ours", "ourselves"}

_WORD = re.compile(r"[a-z0-9']+")


def _count_phrases(text: str, phrases: list[str]) -> int:
    return sum(len(re.findall(rf"\b{re.escape(p)}\b", text)) for p in phrases)


def _mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio (less length-dependent than plain TTR)."""
    if not tokens:
        return np.nan
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([len(set(tokens[i:i + window])) / window
                          for i in range(len(tokens) - window + 1)]))


def linguistic_features(transcript: str, question: str) -> dict[str, float]:
    """Interpretable, observable content/structure features of one response transcript."""
    text = clean_transcript(transcript)
    low = text.lower()
    toks = _WORD.findall(low)
    n = len(toks)
    per100 = (100.0 / n) if n else np.nan
    sents = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    q_toks = set(_WORD.findall(question.lower())) - {"a", "an", "the", "you", "your", "to", "of",
                                                      "in", "and", "or", "is", "are", "what", "how",
                                                      "me", "about", "time", "when", "did", "do"}

    return {
        "empty_transcript": float(n == 0),
        "n_words": float(n),
        "log_n_words": float(np.log1p(n)),
        "n_sentences": float(len(sents)),
        "mean_sentence_words": n / len(sents) if sents else np.nan,
        "mean_word_chars": float(np.mean([len(t) for t in toks])) if toks else np.nan,
        "mattr_50": _mattr(toks),
        "hedge_per100": _count_phrases(low, HEDGES) * per100 if n else np.nan,
        "connective_per100": _count_phrases(low, CONNECTIVES) * per100 if n else np.nan,
        "number_per100": (len(re.findall(r"\b\d+\b", low)) + len(_ONE.findall(low))
                          + _count_phrases(low, NUMBER_WORDS)) * per100 if n else np.nan,
        "fp_singular_per100": sum(t in FP_SING for t in toks) * per100 if n else np.nan,
        "fp_plural_per100": sum(t in FP_PLUR for t in toks) * per100 if n else np.nan,
        "question_overlap": (len(q_toks & set(toks)) / len(q_toks)) if q_toks and n else np.nan,
    }


# --------------------------------------------------------------------------- metrics

def ccc(y: np.ndarray, p: np.ndarray) -> float:
    my, mp = y.mean(), p.mean()
    vy, vp = y.var(), p.var()
    cov = ((y - my) * (p - mp)).mean()
    denom = vy + vp + (my - mp) ** 2
    return float(2 * cov / denom) if denom > 0 else np.nan


def regression_metrics(y, p) -> dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    const = np.allclose(p, p[0])
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(y - p))),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        # 1 - SSE/SST on this split; 0 = no better than predicting this split's mean,
        # negative = worse. Heavy-tailed labels make R2 sensitive to a few extreme responses.
        "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "pearson": np.nan if const else float(stats.pearsonr(y, p)[0]),
        "spearman": np.nan if const else float(stats.spearmanr(y, p)[0]),
        "ccc": ccc(y, p),
    }


def cluster_bootstrap_ci(y, p, groups, metric: str = "spearman", n_boot: int = 1000,
                         seed: int = SEED) -> tuple[float, float]:
    """95% CI resampling participants (not responses), since responses within a person correlate."""
    y, p, groups = np.asarray(y, float), np.asarray(p, float), np.asarray(groups)
    if np.allclose(p, p[0]):
        return (np.nan, np.nan)
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        vals.append(regression_metrics(y[idx], p[idx])[metric])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_diff(y, p_a, p_b, groups, metric: str = "spearman", n_boot: int = 2000,
                          seed: int = SEED) -> tuple[float, float, float]:
    """metric(p_a) - metric(p_b) on the same participant resamples: (point, lo95, hi95).
    Paired differences are much tighter than comparing two separate CIs."""
    y, p_a, p_b, groups = (np.asarray(y, float), np.asarray(p_a, float), np.asarray(p_b, float),
                           np.asarray(groups))
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        d.append(regression_metrics(y[idx], p_a[idx])[metric] - regression_metrics(y[idx], p_b[idx])[metric])
    point = regression_metrics(y, p_a)[metric] - regression_metrics(y, p_b)[metric]
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def within_group_spearman(y, p, groups, min_n: int = 5) -> float:
    """Median Spearman computed separately inside each group (e.g. question), ignoring groups
    with fewer than min_n responses. Removes any credit for recognising the group."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float), "g": np.asarray(groups)})
    vals = [stats.spearmanr(d["y"], d["p"])[0] for _, d in df.groupby("g")
            if len(d) >= min_n and d["p"].nunique() > 1]
    return float(np.nanmedian(vals)) if vals else np.nan


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
