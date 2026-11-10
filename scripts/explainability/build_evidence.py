"""Evidence JSON v0 (Stage 2; finalised in Stage 9).

Builds the delivery evidence for one answer segment from
  - its Stage 1 measurements (segment / sentence / pause tables), and
  - the saved Stage 2 reference distributions (percentile_tables.json).
No training data is touched: the speaker baseline comes from the same speaker's other segments in the tables
you pass in (for a new user, the other answers of the same interview).

The JSON is the only delivery information the feedback LLM will see. Each item keeps the layers apart:
  "measurement": what was measured     {"descriptor": "sentence.speech_rate_wpm", "value": 196.0, "unit": "words/min"}
  "comparison":  how it compares       {"pop_pct": 94.1, "reference": "...", "speaker_median": 148, "self_delta": 0.32}
and there is deliberately no wording: the sentence "your pace was relatively fast..." is Stage 9's job.

Only ACTIVE flags and signals are included. Experimental (pitch, trailing off, uptalk, latency) and disabled
(too_slow) flags are computed into flags.csv for validation but never reach the LLM.

Usage:
  python scripts/explainability/build_evidence.py --clip-id 0001
  python scripts/explainability/build_evidence.py --split val --n 5 --out results/explainability/stage2_reference/evidence_examples
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delivery_reference as dr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import paths as P  # noqa: E402  (results layout)
import descriptor_tables as DT  # noqa: E402  (descriptor tables + Stage 5 columns)
DESC = P.S1
REFERENCE_JSON = P.PERCENTILE_TABLES
MAX_FLAGS = 3
BASELINE_COLUMNS_EXTRA = ["sent_rate_mean_wpm", "mid_pause_max_s", "end_drop_median_db"]


class DeliveryTables:
    """Stage 1 tables for a set of segments, with buckets and speaker baselines attached."""

    def __init__(self, seg: pd.DataFrame, sent: pd.DataFrame, pauses: pd.DataFrame, ref: dr.ReferenceSet):
        seg = seg.copy()
        seg["bucket"] = seg["audio_duration_s"].map(dr.duration_bucket)
        if "duplicate_of" not in seg:
            seg["duplicate_of"] = seg["id"]
        columns = sorted({spec["column"] for spec in ref.refs.values() if spec["level"] == "segment"} | set(BASELINE_COLUMNS_EXTRA))
        self.baselines = pd.DataFrame({c: dr.speaker_baseline(seg, c) for c in columns if c in seg}, index=seg.index)
        self.seg = seg.set_index("id", drop=False)
        self.baselines.index = self.seg.index
        self.sent = {k: g.sort_values("sent_idx").to_dict("records") for k, g in sent.groupby("id")}
        mids = pauses[(pauses["pause_class"] == "mid_sentence") & pauses["counted"]]
        self.mids = {k: g.to_dict("records") for k, g in mids.groupby("id")}
        self.ref = ref
        self._profile = None

    def profile(self) -> pd.DataFrame:
        """Per clip and category: index (0-100), the speaker's usual index (leave-one-out median of their other
        recordings), the difference, and - when the reference set holds fitted thresholds - the hard classes."""
        if self._profile is not None:
            return self._profile
        specs = self.ref.data.get("category_profile", {})
        rows = []
        for cid in self.seg.index:
            ctx = self.context(cid)
            rows.append({"id": cid, **{f"{c}_index": dr.category_index(ctx, self.ref, s)[0] for c, s in specs.items()}})
        prof = pd.DataFrame(rows).set_index("id")
        frame = self.seg[["participant", "duplicate_of"]].join(prof)
        for cat, spec in specs.items():
            prof[f"{cat}_usual"] = dr.speaker_baseline(frame, f"{cat}_index").to_numpy()
            prof[f"{cat}_delta"] = prof[f"{cat}_index"] - prof[f"{cat}_usual"]
            if "thresholds" in spec:
                prof[f"{cat}_level"] = [dr.classify_level(v, spec["thresholds"]) for v in prof[f"{cat}_index"]]
                prof[f"{cat}_class"] = [spec["labels"][lv] if isinstance(lv, str) else None for lv in prof[f"{cat}_level"]]
                prof[f"{cat}_vs_usual"] = [dr.classify_vs_usual(v, spec["vs_usual_thresholds"]) for v in prof[f"{cat}_delta"]]
        self._profile = prof
        return prof

    @classmethod
    def from_results(cls, ref: dr.ReferenceSet) -> "DeliveryTables":
        dtypes = {"id": str, "participant": str, "duplicate_of": str}
        return cls(DT.load_segments(),
                   pd.read_csv(P.SENTENCES, dtype=dtypes),
                   pd.read_csv(P.PAUSES, dtype=dtypes), ref)

    def context(self, clip_id: str) -> dr.ClipContext:
        seg = self.seg.loc[clip_id].to_dict()
        base = {k: v for k, v in self.baselines.loc[clip_id].to_dict().items() if np.isfinite(v)}
        return dr.ClipContext(seg=seg, sentences=self.sent.get(clip_id, []), mid_pauses=self.mids.get(clip_id, []), baseline=base)


def quality_gate_v0(seg: dict) -> dict:
    checks = {"has_speech": bool(np.isfinite(seg.get("speech_span_s", np.nan))),
              "transcript_timing_ok": not bool(seg.get("transcript_timing_suspect", False)),
              "dropouts_below_1pct": bool((seg.get("dropout_frac") or 0) <= 0.01)}
    return {"passed": all(checks.values()), "checks": checks, "version": "v0 (Stage 6 adds a proper quality model)"}


def _clean(obj):
    """JSON-safe: NaN -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k != "magnitude"}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 3)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_evidence(clip_id: str, tables: DeliveryTables, ref: dr.ReferenceSet, question: str | None = None) -> dict:
    ctx = tables.context(clip_id)
    seg = ctx.seg
    flags = dr.all_flags(ctx, ref)
    active = sorted([f for f in flags if f["status"] == "active"], key=lambda f: -f["magnitude"])
    gate = quality_gate_v0(seg)
    sig, cat = dr.scores(ctx, ref)
    categories = {}
    for name, columns in ref.data["categories"].items():
        signals = {}
        for c in columns:
            key = f"segment.{c}"
            v = seg.get(c)
            if v is None or not np.isfinite(v):
                continue
            signals[c] = {"measurement": dr._measurement(ref, key, v), "comparison": dr.compare(ctx, ref, key, v),
                          "signal_score": sig.get(c)}
        categories[name] = {"score": cat.get(name), "signals": signals,
                            "flag_types": [f["type"] for f in active[:MAX_FLAGS] if f["category"] == name]}
    profile = {}
    prof_row = tables.profile().loc[clip_id]
    for cat, spec in ref.data.get("category_profile", {}).items():
        _, parts = dr.category_index(ctx, ref, spec)
        profile[cat] = {
            "title": spec["title"],
            "class": prof_row.get(f"{cat}_class"),
            "level": prof_row.get(f"{cat}_level"),
            "index": prof_row.get(f"{cat}_index"),
            "vs_your_usual": prof_row.get(f"{cat}_vs_usual"),
            "status": spec["status"],
            "use_for_feedback": spec["use_for_feedback"],
            "attention_class": spec["labels"][spec["attention_pole"]] if spec.get("attention_pole") else None,
            "signals": parts,
            "note": spec.get("note"),
        }
    evidence = {
        "segment_id": clip_id,
        "participant": seg.get("participant"),
        "question": question,
        "question_type": seg.get("question_type"),
        "duration_bucket": seg.get("bucket"),
        "predicted_scores": None,  # filled by the score model (WavLM L7 + Ridge) in the full pipeline
        "quality_gate": gate,
        "category_profile": profile,
        "flags": active[:MAX_FLAGS],
        "n_active_flags_total": len(active),
        "categories": categories,
        "strengths": (st := dr.strengths(ctx, ref, sig)),
        "strengths_note": None if st else "no delivery strength with rater support in this answer; take strengths "
                                          "from the content / visual branches",
        "_layers": "measurement = what was measured; comparison = vs reference/speaker; no feedback wording here (Stage 9)",
        "_reference_version": ref.data["version"],
    }
    return _clean(evidence)


def question_lookup() -> dict[str, str]:
    frames = [pd.read_csv(ROOT / "Datasets" / "Split Dataset" / f"{s}.csv", dtype={"id": str})[["id", "question"]]
              for s in ("train", "val", "test")]
    return pd.concat(frames).set_index("id")["question"].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip-id")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ref = dr.ReferenceSet.load(REFERENCE_JSON)
    tables = DeliveryTables.from_results(ref)
    questions = question_lookup()
    if args.clip_id:
        ids = [args.clip_id]
    else:
        pool = tables.seg[(tables.seg["split"] == args.split) & ~tables.seg["is_duplicate"]]
        ids = pool.sample(min(args.n, len(pool)), random_state=args.seed)["id"].tolist()
    for cid in ids:
        ev = build_evidence(cid, tables, ref, questions.get(cid))
        text = json.dumps(ev, indent=2, ensure_ascii=False)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / f"{cid}.json").write_text(text, encoding="utf-8")
        else:
            print(text)


if __name__ == "__main__":
    main()
