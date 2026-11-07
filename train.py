"""Train visual-only skill classifiers, following Rasipuram & Jayagopi (2018):
  correlation analysis -> ReliefF (top-k) -> SVM(RBF) / Logistic Regression
  -> cross-validation, reported against the majority-class baseline.

Differences from the paper (deliberate, to avoid optimistic numbers):
  * ReliefF is fitted INSIDE each CV fold (the paper does not say whether it did this).
  * CV is grouped by person (StratifiedGroupKFold), so one participant never
    appears in both train and test (RecruitView has several videos per user;
    MIT has pre/post interviews per participant).
  * We report balanced accuracy and macro-F1 next to accuracy, plus how stable
    the selected features were across folds.

Edit the CONFIG block, then run:  python train.py
"""
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from skrebate import ReliefF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =============================== CONFIG ======================================
RECRUITVIEW_FEATURES = Path("features_recruitview.csv")
RECRUITVIEW_META = Path("RecruitView2/metadata.jsonl")
MIT_FEATURES = Path("features_mit.csv")
MIT_SCORES = Path("MIT_INTERVIEW_DATASET/Labels/turker_scores_full_interview.csv")

# Confirmed from peek_labels.py output on your data.
RV_TASKS = {                              # task name -> metadata.jsonl column
    "RecruitView Confidence": "confidence_score",
    "RecruitView Engagement (facial expression)": "facial_expression",
    "RecruitView Extraversion": "extraversion",
}
MIT_ID_COL = "Participant"                # values like "p1" / "pp1" (lower-case)
MIT_WORKER_COL = "Worker"                 # "AGGR" rows = EM-aggregated ratings
MIT_TASKS = {                             # task name -> turker_scores column
    "MIT Engaging Tone": "EngagingTone",  # partly vocal; the authors' preferred label
    "MIT Smiled": "Smiled",               # most directly visual
    "MIT Eye Contact": "EyeContact",      # confidence signal in your table
    "MIT Excited": "Excited",
    "MIT Not Awkward": "NotAwkward",      # closest proxy for (low) hesitation
    "MIT Calm": "Calm",
}

# RecruitView quality filter (videos where Py-Feat found a face in too few sampled frames
# give noisy aggregates; see face_rate / n_samples written by the extractor)
MIN_SAMPLES = 8               # min number of 1 Hz samples with a detected face
MIN_FACE_RATE = 0.5           # min fraction of sampled frames with a detected face

TOP_K = 5                     # paper used the top 5 ReliefF features
N_SPLITS = 5
SEED = 0
MODEL_DIR = Path("models")
# =============================================================================


ID_COLS = {"video_id", "file_name", "participant", "phase"}
# Quality indicators: used for filtering only, never as model inputs (a "good video"
# shortcut is not a behavioural feature).
QUALITY_COLS = {"n_samples", "n_frames", "face_rate", "duration_sec", "expected_samples"}


def _norm_name(s: pd.Series) -> pd.Series:
    """'videos/Vid_0001.MP4' -> 'vid_0001'"""
    return (s.astype(str).str.replace("\\", "/", regex=False).str.split("/").str[-1]
             .str.replace(r"\.\w+$", "", regex=True).str.strip().str.lower())


def _digits(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+)")[0], errors="coerce")


def load_recruitview(quality_filter: bool = True) -> pd.DataFrame:
    feats = pd.read_csv(RECRUITVIEW_FEATURES)
    meta = pd.read_json(RECRUITVIEW_META, lines=True)
    # Only the visual features may be used as model inputs. The merged frame also
    # contains every other label (interview_score, overall_performance, ...), which
    # would leak the target if treated as features.
    feat_cols = [c for c in feats.columns if c not in ID_COLS and c not in QUALITY_COLS]

    # Try progressively looser ways to match a video to its metadata row.
    strategies = [
        ("normalised file name", _norm_name(feats["file_name"]), _norm_name(meta["file_name"])),
        ("number inside file name", _digits(feats["file_name"]), _digits(meta["file_name"])),
        ("video number == metadata id", _digits(feats["file_name"]), meta["id"]),
    ]
    best, best_name = None, None
    for name, fk, mk in strategies:
        f2, m2 = feats.assign(_k=fk), meta.assign(_k=mk).drop_duplicates("_k")
        m = f2.merge(m2.drop(columns=["file_name"]), on="_k", how="inner", suffixes=("", "_meta"))
        print(f"  merge by {name}: {len(m)} rows")
        if best is None or len(m) > len(best):
            best, best_name = m, name
        if len(m) >= 0.95 * len(feats):
            break
    df = best
    print(f"RecruitView: {len(feats)} feature rows, {len(meta)} metadata rows -> "
          f"{len(df)} merged (using: {best_name})")
    if len(df) < 0.5 * len(feats):
        print("  !! merge looks wrong. Sample names for comparison:")
        print("     features:", feats["file_name"].head(3).tolist())
        print("     metadata:", meta["file_name"].head(3).tolist(), "| ids:", meta["id"].head(3).tolist())
        raise SystemExit("Fix the file-name match before training.")

    # quality flag. quality_filter=True drops the flagged videos (old behaviour); False keeps them,
    # so every video can still receive a prediction (needed for fusion with the other modalities).
    n0 = len(df)
    ok = pd.Series(True, index=df.index)
    if "n_samples" in df:
        ok &= df["n_samples"] >= MIN_SAMPLES
    if "face_rate" in df:
        ok &= df["face_rate"] >= MIN_FACE_RATE
    df = df.assign(quality_ok=ok)
    if quality_filter:
        df = df[df["quality_ok"]]
    print(f"  quality flag (n_samples>={MIN_SAMPLES}, face_rate>={MIN_FACE_RATE}): "
          f"{int(ok.sum())}/{n0} ok" + ("; others dropped" if quality_filter else "; others kept but flagged"))

    df = df.reset_index(drop=True).copy().assign(group=lambda d: d["user_no"])
    df.attrs["feature_cols"] = feat_cols
    return df


def load_mit() -> pd.DataFrame:
    feats = pd.read_csv(MIT_FEATURES)
    sc = pd.read_csv(MIT_SCORES)
    sc = sc[sc[MIT_WORKER_COL].astype(str).str.upper() == "AGGR"].copy()
    # scores use "p1"/"pp1"; feature files may be "P1"/"PP1" -> compare lower-case
    sc["video_id"] = sc[MIT_ID_COL].astype(str).str.replace(r"\.\w+$", "", regex=True).str.strip().str.lower()
    feat_cols = [c for c in feats.columns if c not in ID_COLS and c not in QUALITY_COLS]
    feats["video_id"] = feats["video_id"].astype(str).str.lower()
    df = feats.merge(sc, on="video_id", how="inner", suffixes=("", "_score"))
    print(f"MIT: {len(feats)} feature rows, {len(sc)} AGGR rows -> {len(df)} merged")
    df = df.copy().assign(group=lambda d: d["participant"])
    df.attrs["feature_cols"] = feat_cols
    return df


def feature_columns(df: pd.DataFrame, label_col: str) -> list:
    """Visual feature columns only (never other labels)."""
    skip_exact = {label_col, "group", "participant", "n_samples", "n_frames"}
    cols = []
    for c in df.attrs["feature_cols"]:
        if c in skip_exact or c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].isna().mean() > 0.3 or df[c].nunique(dropna=True) < 3:
            continue
        cols.append(c)
    return cols


def make_buckets(y: pd.Series, task: str) -> np.ndarray:
    """3cat: low/mid/high by tertiles (paper's '3 Category Task').
       below_avg: bottom third vs rest (paper's 'Below Average Task')."""
    if task == "3cat":
        return pd.qcut(y, 3, labels=False, duplicates="drop").to_numpy()
    cut = y.quantile(1 / 3)
    return (y <= cut).astype(int).to_numpy()


def correlation_report(df, feats, label_col, top=15):
    rows = []
    for c in feats:
        ok = df[[c, label_col]].dropna()
        if len(ok) > 5:
            r, p = spearmanr(ok[c], ok[label_col])
            rows.append((c, r, p))
    out = pd.DataFrame(rows, columns=["feature", "spearman_r", "p"]).dropna()
    out = out.reindex(out.spearman_r.abs().sort_values(ascending=False).index)
    print(f"\nTop {top} features by |Spearman r| with {label_col}:")
    print(out.head(top).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


def build_models():
    return {
        "SVM-RBF": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    }


def relief_select(X, y, k):
    """Return indices of top-k ReliefF features (fit on given data only)."""
    k = min(k, X.shape[1])
    r = ReliefF(n_features_to_select=k, n_neighbors=min(10, max(1, len(y) // 5)), n_jobs=1)
    r.fit(X, y)
    return list(r.top_features_[:k])


def evaluate(df, feats, label_col, task, name):
    d = df.dropna(subset=[label_col]).reset_index(drop=True)
    y = make_buckets(d[label_col], task)
    groups = d["group"].to_numpy()
    X = d[feats].to_numpy(dtype=float)
    counts = Counter(y)
    baseline = max(counts.values()) / len(y)
    print(f"\n=== {name} | task={task} | n={len(y)} | classes={dict(counts)} "
          f"| majority baseline={baseline:.2f}")

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    results = {m: {"acc": [], "bacc": [], "f1": []} for m in build_models()}
    selected = Counter()

    for tr, te in cv.split(X, y, groups):
        pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
        Xtr = pre.fit_transform(X[tr])
        Xte = pre.transform(X[te])
        idx = relief_select(Xtr, y[tr], TOP_K)
        selected.update(feats[i] for i in idx)
        for mname, model in build_models().items():
            model.fit(Xtr[:, idx], y[tr])
            pred = model.predict(Xte[:, idx])
            results[mname]["acc"].append(accuracy_score(y[te], pred))
            results[mname]["bacc"].append(balanced_accuracy_score(y[te], pred))
            results[mname]["f1"].append(f1_score(y[te], pred, average="macro"))

    for mname, r in results.items():
        print(f"  {mname:8s} acc={np.mean(r['acc']):.3f}±{np.std(r['acc']):.3f}  "
              f"bal-acc={np.mean(r['bacc']):.3f}  macro-F1={np.mean(r['f1']):.3f}")
    print("  Feature stability (times selected across folds):")
    for f, n in selected.most_common(8):
        print(f"    {n}/{N_SPLITS}  {f}")
    return d, X, y, results


def fit_and_save(d, X, y, feats, tag):
    """Fit final model on ALL data with ReliefF top-k and save it."""
    pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    Xs = pre.fit_transform(X)
    idx = relief_select(Xs, y, TOP_K)
    model = SVC(kernel="rbf", class_weight="balanced", probability=True).fit(Xs[:, idx], y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"preprocess": pre, "selected": [feats[i] for i in idx],
                 "feature_names": feats, "model": model}, MODEL_DIR / f"{tag}.joblib")
    print(f"  saved {MODEL_DIR / (tag + '.joblib')}  (features: {[feats[i] for i in idx]})")


def run_task(df, label_col, name):
    if label_col is None:
        print(f"\n[skip] {name}: label column not set in CONFIG (run peek_labels.py first)")
        return
    if label_col not in df.columns:
        print(f"\n[skip] {name}: column '{label_col}' not found. Available numeric: "
              f"{df.select_dtypes('number').columns.tolist()[:30]} ...")
        return
    feats = feature_columns(df, label_col)
    print(f"\n{name}: {len(feats)} candidate features")
    correlation_report(df, feats, label_col)
    for task in ("3cat", "below_avg"):
        d, X, y, _ = evaluate(df, feats, label_col, task, name)
    fit_and_save(d, X, make_buckets(d[label_col], "3cat"), feats, name.replace(" ", "_").lower())


def main():
    if RECRUITVIEW_FEATURES.exists() and RECRUITVIEW_META.exists():
        rv = load_recruitview()
        if len(rv) == 0:
            print("RecruitView merge is empty: compare file_name in features vs metadata.")
        for name, col in RV_TASKS.items():
            run_task(rv, col, name)
    else:
        print("RecruitView features/metadata not found; run extract_features_recruitview.py")

    if MIT_FEATURES.exists():
        mit = load_mit()
        if len(mit) < 100:
            print(f"WARNING: only {len(mit)} MIT rows merged (expected 138). "
                  "Check video_id vs Participant naming.")
        for name, col in MIT_TASKS.items():
            run_task(mit, col, name)
    else:
        print("MIT features not found; run extract_features_mit.py")


if __name__ == "__main__":
    main()
