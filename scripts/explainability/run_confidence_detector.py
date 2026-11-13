"""Stage 8, detector 1: per-answer confidence from the legacy WavLM score model.

Hugging Face has no usable "confidence from speech" model (Nov 2026 search). We already have one: WavLM-base-plus
layer 7 mean+std pooled embeddings + Ridge, trained on RecruitView's human `confidence_score` ratings (the best
legacy audio model, rho ~0.45 in repeated CV). Here it is used as a detector:

  RecruitView train   out-of-fold predictions (GroupKFold over participants, 5 folds), so no answer is scored by a
                      model that saw it (or its byte-identical duplicates, which share a participant group here)
  RecruitView val/test predictions of one model fitted on all train participants
  MIT answers         same model; MIT embeddings computed with the legacy extractor (identical preprocessing)

Fitted on train participants only, unique recordings only (duplicate copies averaged). Alpha chosen by grouped CV.
Also predicts overall_performance with the same recipe (used as a secondary check, not for feedback wording).

Output  stage8_detectors/tables/confidence_predictions.csv  id, dataset, split, participant, target, prediction, actual
Run:    python scripts/explainability/run_confidence_detector.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

LAYER = 7
ALPHAS = [100.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
TARGETS = ["confidence_score", "overall_performance"]
EMB_DIR = P.ROOT / "outputs" / "audio" / "cache" / "embeddings" / "wavlm" / "microsoft_wavlm-base-plus"
MIT_EMB_DIR = P.CACHE_S8 / "wavlm_mit"
SPLIT_DIR = P.ROOT / "Datasets" / "Split Dataset"


def recruitview() -> pd.DataFrame:
    seg = pd.read_csv(P.SEGMENTS, dtype={"id": str, "duplicate_of": str},
                      usecols=["id", "split", "participant", "is_duplicate", "duplicate_of"])
    scores = pd.concat([pd.read_csv(SPLIT_DIR / f"{s}.csv", dtype={"id": str}, usecols=["id", *TARGETS])
                        for s in ("train", "val", "test")])
    return seg.merge(scores, on="id", how="left")


def rv_embeddings(df: pd.DataFrame) -> np.ndarray:
    return np.stack([np.load(EMB_DIR / s / f"{cid}.npz")["pooled"][LAYER] for cid, s in zip(df["id"], df["split"])])


def mit_embeddings() -> tuple[pd.DataFrame, np.ndarray]:
    answers = pd.read_csv(P.S7_TABLES / "mit_answers.csv", dtype={"id": str})
    MIT_EMB_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in answers.itertuples() if not (MIT_EMB_DIR / f"{r.id}.npz").exists()]
    if todo:
        sys.path.insert(0, str(P.ROOT / "scripts" / "legacy"))
        import extract_audio_features as legacy  # noqa: E402  (same extractor that built the RecruitView cache)
        config = legacy.load_config()
        extractor = legacy.TransformerEmbeddingExtractor(
            "wavlm", config, legacy.project_path(config["paths"]["cache_dir"]) / "huggingface", allow_download=False)
        for k, r in enumerate(todo, 1):
            signal, sr = legacy.read_wav(Path(r.audio_path))
            extractor.extract(signal, sr, MIT_EMB_DIR / f"{r.id}.npz", use_cache=False)
            if k % 50 == 0 or k == len(todo):
                print(f"  MIT embeddings {k}/{len(todo)}", flush=True)
    X = np.stack([np.load(MIT_EMB_DIR / f"{cid}.npz")["pooled"][LAYER] for cid in answers["id"]])
    return answers, X


def model(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def grouped_oof(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        oof[te] = model(alpha).fit(X[tr], y[tr]).predict(X[te])
    return oof


def main() -> None:
    P.S8_TABLES.mkdir(parents=True, exist_ok=True)
    rv = recruitview()
    X_all = rv_embeddings(rv)
    mit, X_mit = mit_embeddings()

    # Unique train recordings; duplicate copies share the averaged score of their group.
    train = rv[(rv["split"] == "train")]
    uniq = train[~train["is_duplicate"]].copy()
    rows = []
    for target in TARGETS:
        y_group = train.groupby("duplicate_of")[target].mean()
        y = uniq["duplicate_of"].map(y_group).to_numpy(float)
        Xu = X_all[uniq.index.to_numpy()]
        groups = uniq["participant"].to_numpy()
        cv = {a: spearmanr(grouped_oof(Xu, y, groups, a), y)[0] for a in ALPHAS}
        alpha = max(cv, key=cv.get)
        print(f"{target}: grouped-CV Spearman by alpha " + ", ".join(f"{a:g}={v:.3f}" for a, v in cv.items())
              + f" -> alpha {alpha:g}")
        oof = pd.Series(grouped_oof(Xu, y, groups, alpha), index=uniq["duplicate_of"].to_numpy())
        final = model(alpha).fit(Xu, y)

        pred_rv = final.predict(X_all)
        is_train = (rv["split"] == "train").to_numpy()
        pred_rv[is_train] = rv.loc[is_train, "duplicate_of"].map(oof).to_numpy(float)   # OOF for train
        for split in ("train", "val", "test"):
            m = (rv["split"] == split).to_numpy() & rv[target].notna().to_numpy()
            print(f"  {split:5s} n={m.sum():4d}  Spearman {spearmanr(pred_rv[m], rv.loc[m, target])[0]:.3f}")
        rows.append(pd.DataFrame({"id": rv["id"], "dataset": "recruitview", "split": rv["split"],
                                  "participant": rv["participant"].astype(str), "target": target,
                                  "prediction": pred_rv, "actual": rv[target], "alpha": alpha}))
        rows.append(pd.DataFrame({"id": mit["id"], "dataset": "mit", "split": "mit",
                                  "participant": mit["interview"], "target": target,
                                  "prediction": final.predict(X_mit), "actual": np.nan, "alpha": alpha}))
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(P.CONFIDENCE_PRED, index=False)
    print(f"wrote {P.CONFIDENCE_PRED}: {len(out)} rows")


if __name__ == "__main__":
    main()
