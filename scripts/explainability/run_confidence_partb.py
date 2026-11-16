"""Stage 11 (Phase 3, Part B): retrain vocal confidence on RecruitView with the confounds removed from the target,
then test on MIT. Implements the Part B addendum of stage11_style_tagger/preregistration.md (written before running).

Target   residual of confidence_score ~ speech_span_s + log(speech_span_s) + quality_composite, OLS fitted on train
         people only (unique recordings, duplicate copies averaged); the same coefficients give val / test residuals.
B1       WavLM-base-plus layer 7 (mean + std pooled, legacy cache) -> StandardScaler + Ridge, alpha by grouped CV
B2       same features -> logistic regression on the top / bottom 30% of the residual (middle dropped), C by grouped
         CV; score = decision value
RecruitView metric  Spearman(out-of-fold score, residual) over all held-out train answers (5-fold GroupKFold over
         participants); val / test Spearman from one model fitted on all train people
MIT      the all-train model scores every answer; interview means vs the 9-rater mean of Overall, RecommendHiring,
         NotAwkward (+), partial = length + quality + vocal activation; 2,000 bootstrap resamples over students
Pass     CV Spearman >= 0.30 AND no MIT test "contradicted"; promoted when all three are "supported"
B3       (--features b3, after extract_partb_embeddings.py) the same two heads on WavLM-large (layer 6 / 12 / 18 / 24,
         chosen by grouped CV of the ridge head on train people only) and on the audeering emotion model's pooled
         hidden state; per answer mean + std over 10-s windows

Output  stage11_style_tagger/partb_report.md, tables/partb_cv.csv, tables/partb_mit.csv, tables/partb_predictions.csv
Run:    python scripts/explainability/run_confidence_partb.py [--features legacy|b3] [--n-boot 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import mit_label_reliability as LR  # noqa: E402
import run_confidence_detector as CD  # noqa: E402  (legacy embeddings, RecruitView table)
from validate_categories_mit import boot_ci, corr_pair, verdict  # noqa: E402

ALPHAS = [100.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0]
CS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
EXTREME = 0.30
LABELS = ["Overall", "RecommendHiring", "NotAwkward"]
CONTROLS = ["audio_duration_s", "quality_composite", "activation"]
MIN_CV = 0.30
EMB = P.ROOT / "outputs" / "audio" / "cache" / "stage11_style_tagger" / "embeddings"


def pooled(model: str, ids: pd.Series, layer: int) -> np.ndarray:
    """mean + std (over windows) of one layer; NaN rows for answers without windows."""
    out = np.full((len(ids), 2048), np.nan, np.float32)
    for i, cid in enumerate(ids):
        f = EMB / model / f"{cid}.npz"
        if f.exists():
            with np.load(f) as z:
                out[i] = np.concatenate([z["mean"][layer], z["std"][layer]])
    return out


def confound_design(d: pd.DataFrame) -> np.ndarray:
    s = d["speech_span_s"].to_numpy(float)
    return np.column_stack([np.ones(len(d)), s, np.log(np.clip(s, 1.0, None)), d["quality_composite"].to_numpy(float)])


def make_ridge(a):
    return make_pipeline(StandardScaler(), Ridge(alpha=a))


def make_logit(c):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))


def fit_score(kind: str, hp: float, X: np.ndarray, y: np.ndarray):
    """Fit on (X, y) and return a scoring function."""
    if kind == "ridge":
        m = make_ridge(hp).fit(X, y)
        return m.predict
    lo, hi = np.quantile(y, [EXTREME, 1 - EXTREME])
    keep = (y <= lo) | (y >= hi)
    m = make_logit(hp).fit(X[keep], (y[keep] >= hi).astype(int))
    return m.decision_function


def oof(kind: str, hp: float, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    out = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        out[te] = fit_score(kind, hp, X[tr], y[tr])(X[te])
    return out


def load() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    rv = CD.recruitview()
    span = pd.read_csv(P.S4_SEGMENTS, dtype={"id": str}, usecols=["id", "speech_span_s"]).set_index("id")["speech_span_s"]
    gate = pd.read_csv(P.S6_TABLES / "quality_gate_per_clip.csv", dtype={"id": str}).set_index("id")["quality_composite"]
    rv["speech_span_s"] = rv["id"].map(span)
    rv["quality_composite"] = rv["id"].map(gate).fillna(rv["duplicate_of"].map(gate))
    X_rv = CD.rv_embeddings(rv)
    mit, X_mit = CD.mit_embeddings()
    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str}, low_memory=False).set_index("id")
    seg = pd.read_csv(P.S7_TABLES / "mit_segments.csv", dtype={"id": str}).set_index("id")
    mit["activation"] = mit["id"].map(lv["activation"])
    mit["audio_duration_s"] = mit["id"].map(seg["audio_duration_s"])
    mit["quality_composite"] = mit["id"].map(seg["quality_composite"])
    mit["person"] = mit["id"].map(seg["person"])
    return rv, X_rv, mit, X_mit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--features", choices=["legacy", "b3"], default="legacy")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rv, X_rv, mit, X_mit = load()
    rel = pd.read_csv(P.S7_RELIABILITY).set_index("label")["icc2_k"]

    # residual target, fitted on unique train recordings (duplicate copies averaged)
    train = rv[rv["split"] == "train"]
    uniq = train[~train["is_duplicate"]].copy()
    uniq["y_raw"] = uniq["duplicate_of"].map(train.groupby("duplicate_of")["confidence_score"].mean())
    ok = uniq[["y_raw", "speech_span_s", "quality_composite"]].notna().all(1)
    uniq = uniq[ok]
    beta, *_ = np.linalg.lstsq(confound_design(uniq), uniq["y_raw"].to_numpy(float), rcond=None)
    r2 = 1 - np.var(uniq["y_raw"] - confound_design(uniq) @ beta) / np.var(uniq["y_raw"])
    print(f"confounds explain R^2 = {r2:.3f} of train confidence_score (coefficients {np.round(beta, 4).tolist()})")
    y = (uniq["y_raw"] - confound_design(uniq) @ beta).to_numpy(float)
    groups = uniq["participant"].to_numpy()
    rv_ok = rv[["confidence_score", "speech_span_s", "quality_composite"]].notna().all(1)
    rv["y_resid"] = np.where(rv_ok, rv["confidence_score"] - confound_design(rv.fillna({"speech_span_s": 1.0,
                                                                                       "quality_composite": 0.0})) @ beta,
                             np.nan)

    if args.features == "legacy":
        feats = [("legacy WavLM-base-plus L7", X_rv, X_mit, "B1", "B2")]
    else:
        feats = []
        lay = {}
        for li, layer in enumerate([6, 12, 18, 24]):
            Xr = pooled("wavlm_large", rv["id"], li)
            Xs = Xr[uniq.index.to_numpy()]
            ok_ = np.isfinite(Xs).all(1)
            lay[layer] = max(spearmanr(oof("ridge", a, Xs[ok_], y[ok_], groups[ok_]), y[ok_])[0] for a in ALPHAS)
        best = max(lay, key=lay.get)
        print("WavLM-large layer (grouped CV, ridge, train only): " +
              ", ".join(f"L{k}={v:.3f}" for k, v in lay.items()) + f" -> L{best}", flush=True)
        li = [6, 12, 18, 24].index(best)
        feats.append((f"WavLM-large L{best}", pooled("wavlm_large", rv["id"], li),
                      pooled("wavlm_large", mit["id"], li), "B3", "B3"))
        feats.append(("audeering emotion model", pooled("audeering", rv["id"], 0), pooled("audeering", mit["id"], 0),
                      "B3", "B3"))
    cv_rows, mit_rows, preds = [], [], []
    for fname, X_rv, X_mit, tag1, tag2 in feats:
      Xall_u = X_rv[uniq.index.to_numpy()]
      keep_u = np.isfinite(Xall_u).all(1)
      Xu, yk, gk = Xall_u[keep_u], y[keep_u], groups[keep_u]
      for name, kind, grid in ((f"{tag1} ridge on residual ({fname})", "ridge", ALPHAS),
                               (f"{tag2} extremes logistic ({fname})", "logit", CS)):
        y_, groups_ = yk, gk
        cv = {hp: spearmanr(oof(kind, hp, Xu, y_, groups_), y_)[0] for hp in grid}
        hp = max(cv, key=cv.get)
        score = fit_score(kind, hp, Xu, y_)
        row = {"variant": name, "hyperparameter": hp, "cv_spearman": cv[hp],
               "cv_by_hyperparameter": "; ".join(f"{k:g}={v:.3f}" for k, v in cv.items())}
        for split in ("val", "test"):
            m = (rv["split"] == split) & ~rv["is_duplicate"] & rv["y_resid"].notna() & np.isfinite(X_rv).all(1)
            s = score(X_rv[m.to_numpy()])
            row[f"{split}_spearman"] = spearmanr(s, rv.loc[m, "y_resid"])[0]
            row[f"{split}_spearman_raw_score"] = spearmanr(s, rv.loc[m, "confidence_score"])[0]
        cv_rows.append(row)
        print(f"{name}: CV {cv[hp]:.3f} (hp {hp:g}); val {row['val_spearman']:.3f}, test {row['test_spearman']:.3f}",
              flush=True)

        mit["score"] = np.nan
        okm = np.isfinite(X_mit).all(1)
        mit.loc[okm, "score"] = score(X_mit[okm])
        preds.append(mit[["id", "score"]].assign(variant=name))
        iv = mit.groupby("interview")[["score", *CONTROLS]].mean()
        iv["person"] = mit.groupby("interview")["person"].first()
        s_ = LR.load_scores()
        iv = iv.join(s_[s_["Worker"] != "AGGR"].groupby("Participant")[LR.LABELS].mean(), how="inner").reset_index()
        for lab in LABELS:
            r, rp = corr_pair(iv, "score", lab, CONTROLS)
            lo, hi, plo, phi = boot_ci(iv, "score", lab, CONTROLS, args.n_boot, rng)
            mit_rows.append({"variant": name, "label": lab, "rho": r, "ci_lo": lo, "ci_hi": hi, "partial": rp,
                             "p_lo": plo, "p_hi": phi, "verdict": verdict(+1, lo, hi, plo, phi, rel[lab] >= 0.5)})
            print(f"   MIT {lab:16s} {r:+.2f} [{lo:+.2f},{hi:+.2f}] partial {rp:+.2f} [{plo:+.2f},{phi:+.2f}] "
                  f"{mit_rows[-1]['verdict']}", flush=True)

    cvt, mt = pd.DataFrame(cv_rows), pd.DataFrame(mit_rows)
    sfx = "" if args.features == "legacy" else "_b3"
    res = []
    for r in cvt.itertuples():
        v = mt[mt["variant"] == r.variant]
        not_neg = bool((v["verdict"] != "contradicted").all())
        passed = r.cv_spearman >= MIN_CV and not_neg
        res.append("promoted" if passed and (v["verdict"] == "supported").all() else "passed" if passed else
                   "failed: " + ("RecruitView CV < 0.30" if r.cv_spearman < MIN_CV else "") +
                   (" and " if r.cv_spearman < MIN_CV and not not_neg else "") +
                   ("MIT contradicted" if not not_neg else ""))
    cvt["outcome"] = res

    P.S11_TABLES.mkdir(parents=True, exist_ok=True)
    cvt.to_csv(P.S11_TABLES / f"partb{sfx}_cv.csv", index=False)
    mt.to_csv(P.S11_TABLES / f"partb{sfx}_mit.csv", index=False)
    pd.concat(preds).to_csv(P.S11_TABLES / f"partb{sfx}_predictions.csv", index=False)
    md = ["# Stage 11, Part B: confidence retrained on RecruitView with confounds removed", "",
          "Generated by `run_confidence_partb.py`, following the Part B addendum of `preregistration.md`.", "",
          f"Confounds (speaking time, its log, recording quality) explain R² = {r2:.3f} of train `confidence_score`. "
          "The target is the residual.", "",
          "## RecruitView (target = residual)", "",
          "| variant | hyperparameter | grouped-CV Spearman | val | test | test vs raw score | outcome |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r.variant} | {r.hyperparameter:g} | {r.cv_spearman:.3f} | {r.val_spearman:.3f} | "
           f"{r.test_spearman:.3f} | {r.test_spearman_raw_score:.3f} | **{r.outcome}** |" for r in cvt.itertuples()]
    md += ["", "## MIT (interview level; partial = length, quality, vocal activation)", "",
           "| variant | label | rho [95% CI] | partial [95% CI] | verdict |", "|---|---|---|---|---|"]
    md += [f"| {r.variant} | {r.label} | {r.rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | {r.partial:+.2f} "
           f"[{r.p_lo:+.2f}, {r.p_hi:+.2f}] | {r.verdict} |" for r in mt.itertuples()]
    (P.S11 / f"partb{sfx}_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(cvt[["variant", "cv_spearman", "val_spearman", "test_spearman", "outcome"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
