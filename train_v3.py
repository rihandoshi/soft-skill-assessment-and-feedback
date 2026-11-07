"""
train_v3.py  -  improved VIDEO model for RecruitView on the TEAM split (train/val/test by user_no).

Run (from the project folder, venv active):
    python build_features_v3.py      # once, ~1 min, reads the existing frame cache
    python train_v3.py               # ~2-3 min
    python train_v3.py --no-person-mean    # variant that never looks at a person's other videos

What changed vs train_v2.py and why (all measured on the team split, see results_v3/):
 1. log(video length) as a feature. Length is the strongest single predictor of every RecruitView label
    (test Pearson 0.38-0.47 on its own). train_v2 only had raw seconds, which is linear and much weaker
    (0.32-0.39). Text/voice models get length for free (word count, speech time), so leaving it out
    made the video model look far worse than it is.
 2. New visual features (build_features_v3.py): 52 blendshapes already in the Py-Feat cache (eye-look
    directions -> gaze aversion, blinks, smile, brow, jaw), framing (face size/centre), head-motion energy,
    opening/middle/closing thirds of the answer. Combined with the old aggregate features.
 3. Ensemble of Ridge + gradient boosting + RBF-SVR (z-scored average). Each model alone is noisier.
 4. Train-label winsorising at the 2nd/98th percentile (labels have tails out to +-10 SD).
 5. Person-context features (optional, ON by default): the mean of each feature over the same person's
    other answers. Uses no labels; a person's answers are always in the same split, so no leakage.
    It needs several answers per person at prediction time; switch it off with --no-person-mean.
 6. Two training targets: each label itself ("own"), and a composite of the 7 correlated labels
    ("composite", z-scored with TRAIN stats). The labels are essentially one general factor, so the
    less noisy composite often predicts each single label better.

Reported per target: Pearson r on VAL and TEST (with 95% CI by bootstrapping people), Spearman,
and the partial Pearson on TEST controlling for log length (= what video adds beyond length).
Outputs:
    results_v3/results_v3.csv
    results_v3/preds_v3.csv   (all 2011 ids; train rows = out-of-fold by user, val/test = model fit on train)
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
META = ROOT / "RecruitView2" / "metadata.jsonl"
SPLIT_DIR = ROOT / "RecruitView2" / "Split Dataset"
OUT = ROOT / "results_v3"

TARGETS = ["confidence_score", "speaking_skills", "answer_score", "facial_expression",
           "overall_performance", "extraversion", "interview_score"]
COMPOSITE_LABELS = TARGETS  # all 7 load on one general factor
N_BOOT = 1000
SEED = 0


# ---------------------------------------------------------------- data
def load():
    meta = pd.read_json(META, lines=True)
    meta["file_name"] = meta.file_name.astype(str).str.split("/").str[-1]
    meta["user_no"] = meta.user_no.astype(int)
    split = {}
    for s in ["train", "val", "test"]:
        for u in (SPLIT_DIR / f"{s}_users.txt").read_text().split():
            split[int(u)] = s
    meta["split"] = meta.user_no.map(split)
    assert meta.split.notna().all(), "some users are not in any split file"

    old = pd.read_csv(ROOT / "features_recruitview.csv")
    old["file_name"] = old.file_name.astype(str).str.split("/").str[-1]
    v3 = pd.read_csv(ROOT / "features_recruitview_v3.csv")
    v3.columns = [c if c == "file_name" else "n3_" + c for c in v3.columns]

    df = meta.merge(old, on="file_name", how="left").merge(v3, on="file_name", how="left")
    non = set(meta.columns) | {"video_id", "file_name", "expected_samples", "duration_sec", "n_samples", "face_rate"}
    old_cols = [c for c in old.columns if c not in non]
    v3_cols = [c for c in v3.columns if c != "file_name" and not c.startswith("n3_v3_")]
    df["log_dur"] = np.log1p(df.duration_sec)
    feats = list(dict.fromkeys(v3_cols + old_cols + ["log_dur", "duration_sec"]))
    return df, feats


def add_person_mean(df, feats):
    src = [c for c in feats if c != "duration_sec"]
    pm = df.groupby("user_no")[src].transform("mean")
    pm.columns = ["pm_" + c for c in src]
    return pd.concat([df, pm], axis=1), feats + list(pm.columns)


# ---------------------------------------------------------------- models
def make_models():
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "Ridge": lambda: make_pipeline(imp(), StandardScaler(), RidgeCV(alphas=np.logspace(0, 4.5, 12))),
        "HGB": lambda: make_pipeline(imp(), HistGradientBoostingRegressor(
            learning_rate=0.03, max_iter=400, max_depth=3, min_samples_leaf=30,
            l2_regularization=1.0, max_features=0.3, random_state=SEED)),
        "SVR": lambda: make_pipeline(imp(), StandardScaler(), SVR(C=1.0, epsilon=0.2, gamma="scale")),
    }


def winsor(y, ref):
    lo, hi = np.percentile(ref, [2, 98])
    return np.clip(y, lo, hi)


class Ensemble:
    """z-scored average of Ridge/HGB/SVR; z-scaling uses each model's TRAIN-prediction stats."""

    def fit(self, X, y):
        self.ms, self.st = [], []
        for fn in make_models().values():
            m = fn().fit(X, y)
            p = m.predict(X)
            self.ms.append(m); self.st.append((p.mean(), p.std() + 1e-9))
        return self

    def predict(self, X):
        return np.mean([(m.predict(X) - mu) / s for m, (mu, s) in zip(self.ms, self.st)], axis=0)


def oof_train(X, y, groups, n_splits=5):
    out = np.full(len(y), np.nan)
    for a, b in GroupKFold(n_splits).split(X, y, groups):
        out[b] = Ensemble().fit(X.iloc[a], winsor(y[a], y[a])).predict(X.iloc[b])
    return out


# ---------------------------------------------------------------- metrics
def boot_ci(pred, y, groups, rng):
    users = np.unique(groups)
    idx_by_u = {u: np.where(groups == u)[0] for u in users}
    rs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([idx_by_u[u] for u in rng.choice(users, len(users))])
        rs.append(pearsonr(pred[pick], y[pick])[0])
    return np.percentile(rs, [2.5, 97.5])


def partial_r(pred, y, ctrl):
    X = np.c_[np.ones(len(y)), ctrl]
    res = lambda a: a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    return pearsonr(res(pred), res(y))[0]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-person-mean", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    df, feats = load()
    if not args.no_person_mean:
        df, feats = add_person_mean(df, feats)
    tag = "nopm" if args.no_person_mean else "pm"
    tr, va, te = (df.split == s for s in ["train", "val", "test"])
    print(f"Team split: users train/val/test = "
          f"{df[tr].user_no.nunique()}/{df[va].user_no.nunique()}/{df[te].user_no.nunique()}, "
          f"responses {tr.sum()}/{va.sum()}/{te.sum()}; features {len(feats)}; person-mean {'OFF' if args.no_person_mean else 'ON'}")

    mu, sd = df.loc[tr, COMPOSITE_LABELS].mean(), df.loc[tr, COMPOSITE_LABELS].std()
    df["composite"] = ((df[COMPOSITE_LABELS] - mu) / sd).mean(axis=1)

    X = df[feats]
    preds = df[["id", "file_name", "user_no", "split"]].copy()
    rows = []
    fitted = {}
    for train_on in ["composite"] + TARGETS:
        y = df[train_on].values
        m = Ensemble().fit(X[tr], winsor(y[tr], y[tr]))
        p = np.full(len(df), np.nan)
        p[~tr] = m.predict(X[~tr])
        p[tr.values] = oof_train(X[tr].reset_index(drop=True), y[tr], df.user_no[tr].values)
        preds[f"pred_{train_on}"] = p
        fitted[train_on] = p
        print(f"  fitted target={train_on}", flush=True)

    for t in TARGETS:
        y = df[t].values
        for train_on in ["own", "composite"]:
            p = fitted[t if train_on == "own" else "composite"]
            r = dict(target=t, trained_on=train_on, person_mean=not args.no_person_mean)
            for name, mask in [("val", va.values), ("test", te.values)]:
                r[f"{name}_pearson"] = pearsonr(p[mask], y[mask])[0]
            r["test_spearman"] = spearmanr(p[te], y[te])[0]
            r["test_lo"], r["test_hi"] = boot_ci(p[te.values], y[te.values], df.user_no[te].values, rng)
            r["test_partial_len"] = partial_r(p[te.values], y[te.values], df.log_dur[te].values)
            r["test_logdur_only"] = pearsonr(-df.log_dur[te], y[te])[0]
            rows.append(r)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"results_v3_{tag}.csv", index=False)
    preds.to_csv(OUT / f"preds_v3_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    show = res.copy()
    show["test_95CI"] = show.apply(lambda r: f"[{r.test_lo:.2f}, {r.test_hi:.2f}]", axis=1)
    print("\nPearson r (team split). test_logdur_only = length alone; test_partial_len = beyond length")
    print(show[["target", "trained_on", "val_pearson", "test_pearson", "test_95CI", "test_spearman",
                "test_partial_len", "test_logdur_only"]].round(3).to_string(index=False))
    print(f"\nwrote {OUT / f'results_v3_{tag}.csv'} and {OUT / f'preds_v3_{tag}.csv'}")


if __name__ == "__main__":
    main()
