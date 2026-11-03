"""Build and cache text-branch inputs for every RecruitView response.

Writes to outputs/text/features/:
  responses.parquet          ids, split, question, cleaned answer, targets, linguistic features
  emb_<model>_answer.npy     answer embeddings, row-aligned with responses.parquet
  emb_<model>_question.npy   question embeddings, row-aligned with responses.parquet

Usage:
  python text_branch/build_text_features.py [--embed-model sentence-transformers/all-mpnet-base-v2]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from text_common import ID_COLS, OUT_DIR, SPLITS, TARGETS, linguistic_features, load_splits, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    dfs = load_splits()
    df = pd.concat([dfs[s] for s in SPLITS], ignore_index=True)
    feats = pd.DataFrame([linguistic_features(t, q) for t, q in zip(df["transcript"], df["question"])])

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embed_model)
    model.max_seq_length = 512
    ans = model.encode(df["answer_text"].tolist(), batch_size=args.batch_size,
                       normalize_embeddings=True, show_progress_bar=True)
    q_unique = df["question"].unique().tolist()
    q_emb = dict(zip(q_unique, model.encode(q_unique, normalize_embeddings=True)))
    qst = np.stack([q_emb[q] for q in df["question"]])

    # Semantic relevance of the answer to its question (cosine of normalised embeddings).
    empty = feats["empty_transcript"].to_numpy() == 1
    feats["qa_cosine"] = np.where(empty, np.nan, (ans * qst).sum(1))
    ans[empty] = 0.0

    out = pd.concat([df[ID_COLS + ["split", "question", "answer_text"] + TARGETS], feats], axis=1)
    # Clip length in seconds, for the duration_only diagnostic baseline and the length
    # decomposition only; timing features belong to the audio branch, so it is not a model input.
    wav_dir = OUT_DIR / "wav"
    if wav_dir.exists():
        import soundfile as sf
        out["clip_seconds"] = [sf.info(wav_dir / f"{i}.wav").duration if (wav_dir / f"{i}.wav").exists()
                               else np.nan for i in out["id"]]
    fdir = OUT_DIR / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(fdir / "responses.parquet", index=False)
    tag = args.embed_model.split("/")[-1]
    np.save(fdir / f"emb_{tag}_answer.npy", ans.astype(np.float32))
    np.save(fdir / f"emb_{tag}_question.npy", qst.astype(np.float32))
    write_json(fdir / "build_info.json", {
        "embed_model": args.embed_model, "embedding_dim": int(ans.shape[1]),
        "n_responses": len(out), "split_counts": out["split"].value_counts().to_dict(),
        "n_empty_transcripts": int(empty.sum()),
        "linguistic_features": [c for c in feats.columns],
        "inputs": "question + cleaned transcript; gemini_summary excluded (LLM-written, label leakage risk)",
    })
    print(f"Saved {len(out)} responses, {feats.shape[1]} features, {ans.shape[1]}-d embeddings to {fdir}")


if __name__ == "__main__":
    main()
