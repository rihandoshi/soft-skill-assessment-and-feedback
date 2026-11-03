"""Text + audio fusion on the shared participant split: late fusion and early fusion.

Nothing in either branch is retrained or modified; this reads their saved outputs.

Inputs:
  text   outputs/text/predictions/finetune_deberta-v3-base_v2.parquet (final text model:
         train = 5-fold out-of-fold, val/test = fit on train)
         outputs/text/features/responses.parquet + emb_all-mpnet-base-v2_answer.npy
  audio  outputs/audio_branch_run/outputs/audio/... written by the audio-diagnostics code
         (run through fusion/run_audio_branch.py):
         reports/predictions/<target>_{oof,val,test}_predictions.csv (train = GroupKFold OOF,
         val = fit on train, test = refit on train+val, as compare_audio_models.py does)
         features/legacy/<split>_features.csv, embeddings/wavlm/<split>_embeddings.npz

Late fusion (per target; every weight and stacker is fit on train out-of-fold predictions):
  text, audio   each branch alone
  mean          (text + audio) / 2
  weighted      w*audio + (1-w)*text, w chosen by CCC on train OOF
  stack_pair    Ridge on [text, audio] for the same target
  stack_all     Ridge on every branch prediction (4 text targets + 3 audio targets)
  stack_gbm     gradient-boosted trees on the same 7 predictions (non-linear stacker)
  stack_mlp     small MLP (16 hidden units) on the same 7 predictions
answer_score has no audio prediction, so only text and the stack_all/gbm/mlp stackers apply.

Early fusion (one model on concatenated feature blocks; imputation, scaling and PCA are fit
inside each fold):
  early_text    Ridge: linguistic features + answer embedding (32 PCs)
  early_audio   Ridge: handcrafted audio features + WavLM layer (32 PCs)
  early_concat  Ridge: all four blocks
  early_mlp     small MLP on all four blocks
  hybrid        Ridge: DeBERTa predictions + handcrafted audio + WavLM (32 PCs)

Hyperparameters (Ridge alpha, MLP L2, GBM depth/rate/iterations) are chosen by mean CCC over the text branch's grouped folds (GroupKFold 5, shuffled,
seed 42, by user_no); training targets are winsorised at the 1st/99th percentile, as in the
text branch. Reports validation only unless --eval-test, which should be run once, after
choosing on validation.

Each run is a separate version: outputs/fusion_versions/<tag>/ gets predictions/<method>.parquet,
reports/fusion_<split>.{csv,md}, fusion_config.json and a copy of this script. An existing tag is
never overwritten; --eval-test on an existing tag only adds reports/fusion_test.*.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PE / "text_branch"))
from text_common import (SEED, TARGETS, ccc, cluster_bootstrap_ci, paired_bootstrap_diff,  # noqa: E402
                         regression_metrics)

TEXT_OUT = PE / "outputs" / "text"
AUDIO_OUT = PE / "outputs" / "audio_branch_run" / "outputs" / "audio"
LAYER_FILE = PE / "audio_branch" / "results" / "ssl_layer_sweep" / "selected_layers.json"
OUT_DIR = PE / "outputs" / "fusion_versions"
TEXT_MODEL = "finetune_deberta-v3-base_v2"
AUDIO_TARGETS = ["confidence_score", "speaking_skills", "overall_performance"]
ID_COLS = ["id", "user_no", "question_id", "split"]
LING_COLS = ["empty_transcript", "log_n_words", "n_sentences", "mean_sentence_words",
             "mean_word_chars", "mattr_50", "hedge_per100", "connective_per100", "number_per100",
             "fp_singular_per100", "fp_plural_per100", "question_overlap", "qa_cosine"]
ALPHAS = np.logspace(-2, 5, 15)
N_FOLDS = 5
N_PCA = 32
W_GRID = np.round(np.linspace(0, 1, 21), 2)
MLP_ALPHAS = [0.1, 1.0, 10.0, 30.0, 100.0]
GBM_GRID = [{"max_depth": d, "learning_rate": lr, "max_iter": n, "min_samples_leaf": 40}
            for d in (2, 3) for lr in (0.03, 0.1) for n in (100, 300)]


# --------------------------------------------------------------------------- loading

def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """One row per response (responses.parquet order) with labels, branch predictions and
    early-fusion feature blocks."""
    df = pd.read_parquet(TEXT_OUT / "features" / "responses.parquet")
    text = pd.read_parquet(TEXT_OUT / "predictions" / f"{TEXT_MODEL}.parquet")
    df = df.merge(text[["id"] + [f"pred_{t}" for t in TARGETS]]
                  .rename(columns={f"pred_{t}": f"text_{t}" for t in TARGETS}), on="id", how="left")
    length = pd.read_parquet(TEXT_OUT / "predictions" / "length_only.parquet")
    df = df.merge(length[["id"] + [f"pred_{t}" for t in TARGETS]]
                  .rename(columns={f"pred_{t}": f"length_{t}" for t in TARGETS}), on="id", how="left")

    pdir = AUDIO_OUT / "reports" / "predictions"
    for t in AUDIO_TARGETS:
        parts = [pd.read_csv(pdir / f"{t}_oof_predictions.csv", dtype={"id": str})
                 .rename(columns={"oof_prediction": "prediction"}),
                 pd.read_csv(pdir / f"{t}_val_predictions.csv", dtype={"id": str})]
        if (pdir / f"{t}_test_predictions.csv").is_file():
            parts.append(pd.read_csv(pdir / f"{t}_test_predictions.csv", dtype={"id": str}))
        a = pd.concat(parts)[["id", "prediction"]].rename(columns={"prediction": f"audio_{t}"})
        assert a.id.is_unique, f"duplicate ids in audio predictions for {t}"
        df = df.merge(a, on="id", how="left")

    hand = pd.concat([pd.read_csv(AUDIO_OUT / "features" / "legacy" / f"{s}_features.csv", dtype={"id": str})
                      for s in ("train", "val", "test")])
    hand_cols = [c for c in hand.columns if c not in ("id", "user_id", "question_id", "split",
                 "feature_group", "feature_status", "feature_error")
                 and pd.api.types.is_numeric_dtype(hand[c]) and hand[c].nunique() > 1]
    hand = hand.set_index("id").reindex(df.id)[hand_cols].to_numpy(float)

    layer = json.loads(LAYER_FILE.read_text(encoding="utf-8"))[
        "wavlm/overall_performance/current_mean_std"]["layer_index"]
    wav = np.full((len(df), 0), np.nan)
    pos = {i: k for k, i in enumerate(df.id)}
    for s in ("train", "val", "test"):
        with np.load(AUDIO_OUT / "embeddings" / "wavlm" / f"{s}_embeddings.npz") as z:
            if wav.shape[1] == 0:
                wav = np.full((len(df), z["pooled"].shape[2]), np.nan, np.float32)
            idx = [pos[i] for i in z["ids"].astype(str)]
            wav[idx] = z["pooled"][:, layer, :]

    blocks = {
        "text_ling": df[LING_COLS].to_numpy(float),
        "text_emb": np.load(TEXT_OUT / "features" / "emb_all-mpnet-base-v2_answer.npy"),
        "audio_hand": hand,
        "audio_wavlm": wav,
    }
    print(f"loaded {len(df)} responses; handcrafted audio {hand.shape[1]} cols; "
          f"WavLM layer {layer} ({wav.shape[1]} dims)")
    return df, blocks


# --------------------------------------------------------------------------- models

def winsor(y: np.ndarray) -> np.ndarray:
    return np.clip(y, *np.percentile(y, [1, 99]))


def make_early(block_sizes: dict[str, int], head):
    parts, start = [], 0
    for name, size in block_sizes.items():
        cols = slice(start, start + size)
        start += size
        steps = [SimpleImputer(strategy="median"), StandardScaler()]
        if name in ("text_emb", "audio_wavlm"):
            steps += [PCA(N_PCA, random_state=SEED), StandardScaler()]
        parts.append((name, make_pipeline(*steps), cols))
    return make_pipeline(ColumnTransformer(parts), head)


def ridge(alpha: float):
    return Ridge(alpha=alpha)


def mlp(alpha: float):
    # One small hidden layer; L2 (alpha) is the main regulariser. No internal early stopping,
    # because its random 10% holdout would ignore the participant grouping.
    return MLPRegressor(hidden_layer_sizes=(16,), alpha=alpha, learning_rate_init=1e-3, max_iter=500, random_state=SEED)


def gbm(params: dict):
    return HistGradientBoostingRegressor(**params, l2_regularization=1.0, random_state=SEED)


def make_stacker(head):
    return make_pipeline(StandardScaler(), head)


def fit_family(build, X: np.ndarray, y: np.ndarray, tr: np.ndarray, folds, grid=ALPHAS):
    """Choose the hyperparameter by mean fold CCC on train; return (OOF train preds +
    train-fit preds for the rest, chosen hyperparameter, cv_ccc)."""
    Xtr, ytr = X[tr], y[tr]
    best = (-np.inf, grid[0], None)
    for a in grid:
        oof, scores = np.empty(tr.sum()), []
        for f_tr, f_va in folds:
            oof[f_va] = build(a).fit(Xtr[f_tr], winsor(ytr[f_tr])).predict(Xtr[f_va])
            scores.append(ccc(ytr[f_va], oof[f_va]))
        if np.mean(scores) > best[0]:
            best = (float(np.mean(scores)), a, oof)
    cv, alpha, oof = best
    p = np.empty(len(X))
    p[tr] = oof
    p[~tr] = build(alpha).fit(Xtr, winsor(ytr)).predict(X[~tr])
    return p, alpha, cv


def late_fusion(df: pd.DataFrame, tr: np.ndarray, folds) -> tuple[dict[str, pd.DataFrame], dict]:
    stackers = {"stack_all": (lambda a: make_stacker(ridge(a)), ALPHAS),
                "stack_gbm": (gbm, GBM_GRID),
                "stack_mlp": (lambda a: make_stacker(mlp(a)), MLP_ALPHAS)}
    methods = {m: df[ID_COLS].copy() for m in
               ("text", "audio", "length_only", "mean", "weighted", "stack_pair", *stackers)}
    chosen = {}
    all_inputs = [f"text_{t}" for t in TARGETS] + [f"audio_{t}" for t in AUDIO_TARGETS]
    X_all = df[all_inputs].to_numpy(float)
    for t in TARGETS:
        y = df[t].to_numpy(float)
        methods["text"][f"pred_{t}"] = df[f"text_{t}"]
        methods["length_only"][f"pred_{t}"] = df[f"length_{t}"]
        for name, (build, grid) in stackers.items():
            p, a, cv = fit_family(build, X_all, y, tr, folds, grid)
            methods[name][f"pred_{t}"] = p
            chosen[f"{name}/{t}"] = {"param": a, "cv_ccc": cv}
        if t not in AUDIO_TARGETS:
            continue
        txt, aud = df[f"text_{t}"].to_numpy(float), df[f"audio_{t}"].to_numpy(float)
        methods["audio"][f"pred_{t}"] = aud
        methods["mean"][f"pred_{t}"] = (txt + aud) / 2
        w = max(W_GRID, key=lambda w: ccc(y[tr], (w * aud + (1 - w) * txt)[tr]))
        methods["weighted"][f"pred_{t}"] = w * aud + (1 - w) * txt
        chosen[f"weighted/{t}"] = {"w_audio": float(w),
                                   "train_oof_ccc": ccc(y[tr], (w * aud + (1 - w) * txt)[tr])}
        p, a, cv = fit_family(lambda a: make_stacker(ridge(a)), np.column_stack([txt, aud]), y, tr, folds)
        methods["stack_pair"][f"pred_{t}"] = p
        chosen[f"stack_pair/{t}"] = {"param": a, "cv_ccc": cv}
    return methods, chosen


def early_fusion(df, blocks, tr, folds) -> tuple[dict[str, pd.DataFrame], dict]:
    concat = ["text_ling", "text_emb", "audio_hand", "audio_wavlm"]
    # hybrid: the fine-tuned DeBERTa predictions (train = out-of-fold) stand in for the text
    # features, since the frozen-embedding text block is much weaker than DeBERTa.
    variants = {"early_text": (["text_ling", "text_emb"], ridge, ALPHAS),
                "early_audio": (["audio_hand", "audio_wavlm"], ridge, ALPHAS),
                "early_concat": (concat, ridge, ALPHAS),
                "early_mlp": (concat, mlp, MLP_ALPHAS),
                "hybrid": (["text_pred", "audio_hand", "audio_wavlm"], ridge, ALPHAS)}
    blocks = {**blocks, "text_pred": df[[f"text_{t}" for t in TARGETS]].to_numpy(float)}
    methods, chosen = {}, {}
    for name, (use, head, grid) in variants.items():
        X = np.hstack([blocks[b] for b in use])
        sizes = {b: blocks[b].shape[1] for b in use}
        out = df[ID_COLS].copy()
        for t in TARGETS:
            p, a, cv = fit_family(lambda a: make_early(sizes, head(a)), X, df[t].to_numpy(float),
                                  tr, folds, grid)
            out[f"pred_{t}"] = p
            chosen[f"{name}/{t}"] = {"param": a, "cv_ccc": cv}
        methods[name] = out
        print(f"done: {name}", flush=True)
    return methods, chosen


# --------------------------------------------------------------------------- evaluation

def partial_spearman(y, p, z) -> float:
    """Spearman of label and prediction after removing a linear fit on log word count from both ranks."""
    ry, rp, rz = (stats.rankdata(v) for v in (y, p, z))
    res = lambda r: r - np.polyval(np.polyfit(rz, r, 1), rz)
    return float(stats.pearsonr(res(ry), res(rp))[0])


def evaluate(df: pd.DataFrame, methods: dict[str, pd.DataFrame], split: str) -> pd.DataFrame:
    m = (df["split"] == split).to_numpy()
    rows = []
    for t in TARGETS:
        y = df.loc[m, t].to_numpy(float)
        g = df.loc[m, "user_no"].to_numpy()
        z = df.loc[m, "log_n_words"].to_numpy(float)
        refs = {"text": methods["text"].loc[m, f"pred_{t}"].to_numpy(float)}
        if t in AUDIO_TARGETS:
            refs["audio"] = methods["audio"].loc[m, f"pred_{t}"].to_numpy(float)
        for name, preds in methods.items():
            if f"pred_{t}" not in preds or preds[f"pred_{t}"].isna().all():
                continue
            p = preds.loc[m, f"pred_{t}"].to_numpy(float)
            ok = np.isfinite(p)
            r = {"method": name, "target": t, "split": split, **regression_metrics(y[ok], p[ok])}
            r["spearman_ci95"] = cluster_bootstrap_ci(y[ok], p[ok], g[ok], "spearman")
            r["ccc_ci95"] = cluster_bootstrap_ci(y[ok], p[ok], g[ok], "ccc")
            r["partial_spearman_len"] = partial_spearman(y[ok], p[ok], z[ok])
            for ref_name, ref in refs.items():
                if name == ref_name:
                    continue
                for metric in ("spearman", "ccc"):
                    d, lo, hi = paired_bootstrap_diff(y[ok], p[ok], ref[ok], g[ok], metric)
                    r[f"d_{metric}_vs_{ref_name}"] = d
                    r[f"d_{metric}_vs_{ref_name}_ci95"] = (lo, hi)
            rows.append(r)
    return pd.DataFrame(rows)


def complementarity(df: pd.DataFrame, tr: np.ndarray) -> pd.DataFrame:
    """Train OOF: how similar the two branches' predictions are, and how much each tracks length."""
    rows = []
    for t in AUDIO_TARGETS:
        d = df.loc[tr]
        rows.append({"target": t,
                     "spearman(text, audio)": stats.spearmanr(d[f"text_{t}"], d[f"audio_{t}"])[0],
                     "spearman(text, words)": stats.spearmanr(d[f"text_{t}"], d["log_n_words"])[0],
                     "spearman(audio, words)": stats.spearmanr(d[f"audio_{t}"], d["log_n_words"])[0],
                     "spearman(audio, seconds)": stats.spearmanr(d[f"audio_{t}"], d["clip_seconds"])[0]})
    return pd.DataFrame(rows)


def fmt_ci(ci) -> str:
    return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if isinstance(ci, tuple) and np.isfinite(ci[0]) else ""


def write_markdown(res: pd.DataFrame, comp: pd.DataFrame, chosen: dict, split: str, path: Path) -> None:
    md = [f"# Text + audio fusion ({split}, n={int(res['n'].max())} responses)", "",
          "Spearman and CCC with 95% participant-bootstrap CIs. d = method minus text-only (or "
          "audio-only) on paired participant resamples (an interval containing 0 means no "
          "detectable difference). "
          "partial | len = Spearman after removing log word count from label and prediction.", "",
          "## Branch complementarity (train out-of-fold)", "",
          "| " + " | ".join(comp.columns) + " |", "|" + "---|" * len(comp.columns),
          *("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
            for row in comp.itertuples(index=False)), ""]
    for t in TARGETS:
        sub = res[res.target == t]
        refs = ["text", "audio"] if t in AUDIO_TARGETS else ["text"]
        cols = [f"d {m} vs {ref}" for ref in refs for m in ("Spearman", "CCC")]
        md += [f"## {t}", "", "| method | Spearman | CCC | RMSE | partial \\| len | " + " | ".join(cols) + " |",
               "|" + "---|" * (5 + len(cols))]
        for r in sub.to_dict("records"):
            ds = [f"{r[f'd_{m}_vs_{ref}']:+.3f} {fmt_ci(r[f'd_{m}_vs_{ref}_ci95'])}"
                  if r["method"] != ref else "" for ref in refs for m in ("spearman", "ccc")]
            md.append(f"| {r['method']} | {r['spearman']:.3f} {fmt_ci(r['spearman_ci95'])} | {r['ccc']:.3f} "
                      f"{fmt_ci(r['ccc_ci95'])} | {r['rmse']:.3f} | {r['partial_spearman_len']:.3f} | "
                      + " | ".join(ds) + " |")
        w = chosen.get(f"weighted/{t}")
        if w:
            md += ["", f"Weighted fusion: w_audio = {w['w_audio']:.2f} (chosen on train OOF)."]
        md.append("")
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="version name, e.g. v2_nonlinear; never overwritten")
    ap.add_argument("--eval-test", action="store_true",
                    help="also report test metrics; run once, after choosing on validation")
    args = ap.parse_args()
    out = OUT_DIR / args.tag
    pdir, rdir = out / "predictions", out / "reports"
    existing = out.exists()
    if existing and not args.eval_test:
        raise SystemExit(f"{out} already exists; pick a new --tag (versions are never overwritten)")
    if existing and (rdir / "fusion_test.csv").exists():
        raise SystemExit(f"{rdir / 'fusion_test.csv'} already exists; test is evaluated once per version")

    df, blocks = load_inputs()
    tr = (df["split"] == "train").to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
                 .split(np.zeros(tr.sum()), groups=df.loc[tr, "user_no"].to_numpy()))
    for c in [f"text_{t}" for t in TARGETS] + [f"audio_{t}" for t in AUDIO_TARGETS]:
        assert df.loc[tr | (df.split == "val").to_numpy(), c].notna().all(), f"missing {c} predictions"

    late, chosen = late_fusion(df, tr, folds)
    print("done: late fusion", flush=True)
    early, chosen_early = early_fusion(df, blocks, tr, folds)
    methods, chosen = {**late, **early}, {**chosen, **chosen_early}

    comp = complementarity(df, tr)
    print(comp.round(3).to_string(index=False))
    if existing:  # --eval-test on a finished version: add the test report, touch nothing else
        splits = ["test"]
    else:
        pdir.mkdir(parents=True)
        rdir.mkdir(parents=True)
        for name, preds in methods.items():
            preds.to_parquet(pdir / f"{name}.parquet", index=False)
        (rdir / "fusion_config.json").write_text(json.dumps(
            {"tag": args.tag, "text_model": TEXT_MODEL,
             "folds": "GroupKFold(5, shuffle, seed 42) by user_no on train",
             "n_pca": N_PCA, "chosen": chosen}, indent=2, default=float), encoding="utf-8")
        shutil.copyfile(__file__, out / "fuse.py")
        splits = ["val"] + (["test"] if args.eval_test else [])
    for s in splits:
        res = evaluate(df, methods, s)
        res.to_csv(rdir / f"fusion_{s}.csv", index=False)
        write_markdown(res, comp, chosen, s, rdir / f"fusion_{s}.md")
        print(res[["method", "target", "spearman", "ccc", "rmse", "partial_spearman_len"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
