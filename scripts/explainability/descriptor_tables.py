"""The descriptor tables every comparison / validation stage reads (Stages 2, 3, evidence JSON).

  base tables   paths.SEGMENTS / SENTENCES / PAUSES / REGISTRY (Stage 4 ASR tables, or Stage 1 transcript tables)
  + Stage 5     answer-level SER columns (stage5_ser/tables/segment_ser.csv) and their registry entries, if present

Measurement stays in each stage's own files; this module only joins them, so re-running one measurement stage
never requires re-running another.
"""
from __future__ import annotations

import json

import pandas as pd

import paths as P

DTYPES = {"id": str, "participant": str, "duplicate_of": str}


def load_segments() -> pd.DataFrame:
    seg = pd.read_csv(P.SEGMENTS, dtype=DTYPES)
    if P.S5_SEGMENTS.exists():
        ser = pd.read_csv(P.S5_SEGMENTS, dtype={"id": str})
        seg = seg.merge(ser[[c for c in ser.columns if c == "id" or c not in seg.columns]], on="id", how="left")
    return seg


def load_sentences() -> pd.DataFrame:
    return pd.read_csv(P.SENTENCES, dtype=DTYPES)


def load_pauses() -> pd.DataFrame:
    return pd.read_csv(P.PAUSES, dtype=DTYPES)


def load_registry() -> dict:
    reg = dict(json.loads(P.REGISTRY.read_text(encoding="utf-8"))["descriptors"])
    if P.S5_REGISTRY.exists() and P.S5_SEGMENTS.exists():
        reg.update(json.loads(P.S5_REGISTRY.read_text(encoding="utf-8"))["descriptors"])
    return reg
