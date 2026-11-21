"""Text + audio + video fusion on the shared participant split.

Builds on fuse.py (text + audio, v2), whose loaders, model builders and fit_family are reused
unchanged. Video predictions and features come from fusion/run_video_branch.py (run it first).
Every weight and stacker is fit on train out-of-fold predictions; hyperparameters are chosen by
mean CCC over the same GroupKFold(5, shuffle, seed 42) by user_no folds as fuse.py.

Branches per target: confidence_score, speaking_skills, overall_performance have text, audio and
video; answer_score has text and video only (the audio branch does not predict it).

Missing video: the video teammate's quality filter drops ~5% of answers (too few detected faces).
  mean / rank_mean / weighted  average over the branches that are available (weights renormalised)
  Ridge / MLP stackers         missing video prediction filled with its train mean + a 0/1
                               `video_missing` input, so the stacker learns the fallback
  stack_gbm                    trees handle the NaN natively
  early fusion                 median imputation inside the fold, as for any missing feature

Late fusion (combine predictions):
  text, audio, video      each branch alone (references)
  mean_ta / mean_tv / mean_av, weighted_ta / weighted_tv / weighted_av    two-branch pairs
  mean_3                  plain average of the three branches
  rank_mean_3             average of each branch's percentile within its train OOF predictions,
                          mapped back to the label scale with the train label quantiles
                          (scale-free: a branch with shrunk predictions still counts fully)
  weighted_3              w_t*text + w_a*audio + w_v*video, weights on a 0.05 simplex grid
                          chosen by CCC on train OOF
  stack_3                 Ridge on the three predictions for the same target
  stack_len_3             stack_3 + log word count, to see what fusion adds beyond length
  stack_all_3             Ridge on all 11 branch predictions (4 text + 3 audio + 4 video)
  stack_gbm_3 / stack_mlp_3   gradient-boosted trees / small MLP on the same 11 predictions

Early fusion (combine features; imputation, scaling and PCA fit inside each fold):
  early_video             Ridge on the video teammate's robust visual features
  early_concat_3          Ridge on text linguistic + text embedding (32 PCs) + audio handcrafted
                          + WavLM (32 PCs) + video features
  early_mlp_3             small MLP on the same five blocks
  hybrid_3                Ridge on DeBERTa predictions + audio handcrafted + WavLM + video features

Reports validation only unless --eval-test, which should be run once, after choosing on
validation. Each run is a separate version under outputs/fusion_versions/<tag>/ (never
overwritten), with copies of this script, fuse.py and run_video_branch.py.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

FUSION = Path(__file__).resolve().parent
sys.path.insert(0, str(FUSION))
import fuse as F  # noqa: E402
from run_video_branch import WORK as VIDEO_OUT, load_video_features  # noqa: E402
from text_common import (SEED, TARGETS, ccc, cluster_bootstrap_ci, paired_bootstrap_diff,  # noqa: E402
                         regression_metrics)

SHARED = F.AUDIO_TARGETS
STEP = 0.05
REFS = ("text", "weighted_ta")


def branches(t: str) -> list[str]:
    return ["text", "audio", "video"] if t in SHARED else ["text", "video"]


# --------------------------------------------------------------------------- loading

def load_inputs() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df, blocks = F.load_inputs()
    v = pd.read_parquet(VIDEO_OUT / "predictions.parquet")
    df = df.merge(v[["id"] + [f"pred_{t}" for t in TARGETS]]
                  .rename(columns={f"pred_{t}": f"video_{t}" for t in TARGETS}), on="id", how="left")
    blocks["video"], feats, _ = load_video_features(df["id"])
    df["has_video"] = df[f"video_{TARGETS[0]}"].notna()
    print(f"video: {len(feats)} features, predictions for {df.has_video.sum()}/{len(df)} responses")
    return df, blocks


# --------------------------------------------------------------------------- late fusion

def simplex(k: int) -> list[np.ndarray]:
    grid = np.round(np.arange(0, 1 + 1e-9, STEP), 2)
    return [np.array(w) for w in product(grid, repeat=k) if abs(sum(w) - 1) < 1e-9]


def weighted(P: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted average over the available branches (weights renormalised per row)."""
    avail = np.isfinite(P)
    num = np.where(avail, P, 0) @ w
    den = avail @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nanmean(P, axis=1))


def rank_mean(P: np.ndarray, y: np.ndarray, tr: np.ndarray) -> np.ndarray:
    pct = np.full(P.shape, np.nan)
    for j in range(P.shape[1]):
        ref = np.sort(P[tr, j][np.isfinite(P[tr, j])])
        ok = np.isfinite(P[:, j])
        mid = (np.searchsorted(ref, P[ok, j], "left") + np.searchsorted(ref, P[ok, j], "right")) / 2
        pct[ok, j] = np.clip(mid / len(ref), 0.5 / len(ref), 1 - 0.5 / len(ref))
    return np.quantile(F.winsor(y[tr]), np.nanmean(pct, axis=1))


def stack_inputs(df: pd.DataFrame, cols: list[str], tr: np.ndarray, fill: bool) -> np.ndarray:
    X = df[cols].to_numpy(float)
    if not fill:
        return X
    vcols = [j for j, c in enumerate(cols) if c.startswith("video_")]
    for j in vcols:
        X[np.isnan(X[:, j]), j] = np.nanmean(X[tr, j])
    return np.column_stack([X, (~df["has_video"]).to_numpy(float)]) if vcols else X


def late_fusion(df: pd.DataFrame, tr: np.ndarray, folds) -> tuple[dict[str, pd.DataFrame], dict]:
    all_cols = ([f"text_{t}" for t in TARGETS] + [f"audio_{t}" for t in SHARED]
                + [f"video_{t}" for t in TARGETS])
    stackers = {"stack_all_3": (lambda a: F.make_stacker(F.ridge(a)), F.ALPHAS, True),
                "stack_gbm_3": (F.gbm, F.GBM_GRID, False),
                "stack_mlp_3": (lambda a: F.make_stacker(F.mlp(a)), F.MLP_ALPHAS, True)}
    methods: dict[str, pd.DataFrame] = {}
    chosen: dict = {}

    def put(name, t, p):
        methods.setdefault(name, df[F.ID_COLS].copy())[f"pred_{t}"] = p

    for t in TARGETS:
        y = df[t].to_numpy(float)
        bs = branches(t)
        for b in bs:
            put(b, t, df[f"{b}_{t}"].to_numpy(float))
        put("length_only", t, df[f"length_{t}"].to_numpy(float))

        pairs = [("text", "audio"), ("text", "video"), ("audio", "video")]
        for combo in [p for p in pairs if set(p) <= set(bs)] + [tuple(bs)]:
            P = df[[f"{b}_{t}" for b in combo]].to_numpy(float)
            tag = "3" if len(combo) == 3 else "".join(b[0] for b in combo)
            put(f"mean_{tag}", t, np.nanmean(P, axis=1))
            w = max(simplex(len(combo)), key=lambda w: ccc(y[tr], weighted(P[tr], w)))
            put(f"weighted_{tag}", t, weighted(P, w))
            chosen[f"weighted_{tag}/{t}"] = {**{f"w_{b}": float(x) for b, x in zip(combo, w)},
                                             "train_oof_ccc": ccc(y[tr], weighted(P[tr], w))}
            if len(combo) == len(bs):
                put(f"rank_mean_{tag}", t, rank_mean(P, y, tr))

        same = [f"{b}_{t}" for b in bs]
        for name, cols in {"stack_3": same, "stack_len_3": same + ["log_n_words"]}.items():
            p, a, cv = F.fit_family(lambda a: F.make_stacker(F.ridge(a)),
                                    stack_inputs(df, cols, tr, True), y, tr, folds)
            put(name, t, p)
            chosen[f"{name}/{t}"] = {"param": a, "cv_ccc": cv}
        for name, (build, grid, fill) in stackers.items():
            p, a, cv = F.fit_family(build, stack_inputs(df, all_cols, tr, fill), y, tr, folds, grid)
            put(name, t, p)
            chosen[f"{name}/{t}"] = {"param": a, "cv_ccc": cv}
        print(f"done: late fusion {t}", flush=True)
    return methods, chosen


def early_fusion(df, blocks, tr, folds) -> tuple[dict[str, pd.DataFrame], dict]:
    five = ["text_ling", "text_emb", "audio_hand", "audio_wavlm", "video"]
    variants = {"early_video": (["video"], F.ridge, F.ALPHAS),
                "early_concat_3": (five, F.ridge, F.ALPHAS),
                "early_mlp_3": (five, F.mlp, F.MLP_ALPHAS),
                "hybrid_3": (["text_pred", "audio_hand", "audio_wavlm", "video"], F.ridge, F.ALPHAS)}
    blocks = {**blocks, "text_pred": df[[f"text_{t}" for t in TARGETS]].to_numpy(float)}
    methods, chosen = {}, {}
    for name, (use, head, grid) in variants.items():
        X = np.hstack([blocks[b] for b in use])
        sizes = {b: blocks[b].shape[1] for b in use}
        out = df[F.ID_COLS].copy()
        for t in TARGETS:
            p, a, cv = F.fit_family(lambda a: F.make_early(sizes, head(a)), X, df[t].to_numpy(float),
                                    tr, folds, grid)
            out[f"pred_{t}"] = p
            chosen[f"{name}/{t}"] = {"param": a, "cv_ccc": cv}
        methods[name] = out
        print(f"done: {name}", flush=True)
    return methods, chosen


# --------------------------------------------------------------------------- evaluation

def evaluate(df: pd.DataFrame, methods: dict[str, pd.DataFrame], split: str, video_only: bool) -> pd.DataFrame:
    m = (df["split"] == split).to_numpy() & (df["has_video"].to_numpy() if video_only else True)
    rows = []
    for t in TARGETS:
        y = df.loc[m, t].to_numpy(float)
        g = df.loc[m, "user_no"].to_numpy()
        z = df.loc[m, "log_n_words"].to_numpy(float)
        # the video-only subset is a robustness check: point metrics + one paired comparison
        refs = {r: methods[r].loc[m, f"pred_{t}"].to_numpy(float)
                for r in (("weighted_ta",) if video_only else REFS) if f"pred_{t}" in methods.get(r, {})}
        for name, preds in methods.items():
            if f"pred_{t}" not in preds:
                continue
            p = preds.loc[m, f"pred_{t}"].to_numpy(float)
            ok = np.isfinite(p)
            r = {"method": name, "target": t, "split": split,
                 "subset": "with_video" if video_only else "all", **regression_metrics(y[ok], p[ok])}
            if not video_only:
                r["spearman_ci95"] = cluster_bootstrap_ci(y[ok], p[ok], g[ok], "spearman")
                r["ccc_ci95"] = cluster_bootstrap_ci(y[ok], p[ok], g[ok], "ccc")
            r["partial_spearman_len"] = F.partial_spearman(y[ok], p[ok], z[ok])
            for ref_name, ref in refs.items():
                both = ok & np.isfinite(ref)
                if name == ref_name or not both.any():
                    continue
                for metric in (("spearman",) if video_only else ("spearman", "ccc")):
                    d, lo, hi = paired_bootstrap_diff(y[both], p[both], ref[both], g[both], metric)
                    r[f"d_{metric}_vs_{ref_name}"] = d
                    r[f"d_{metric}_vs_{ref_name}_ci95"] = (lo, hi)
            rows.append(r)
    return pd.DataFrame(rows)


def complementarity(df: pd.DataFrame, tr: np.ndarray) -> pd.DataFrame:
    """Train OOF: how similar the branches' predictions are, and how much each tracks length."""
    rows = []
    d = df.loc[tr & df["has_video"].to_numpy()]
    for t in TARGETS:
        r = {"target": t}
        bs = branches(t)
        for i, a in enumerate(bs):
            for b in bs[i + 1:]:
                r[f"{a}~{b}"] = stats.spearmanr(d[f"{a}_{t}"], d[f"{b}_{t}"])[0]
        for b in bs:
            r[f"{b}~words"] = stats.spearmanr(d[f"{b}_{t}"], d["log_n_words"])[0]
        rows.append(r)
    return pd.DataFrame(rows)


def write_markdown(res: pd.DataFrame, comp: pd.DataFrame, chosen: dict, split: str, path: Path) -> None:
    allv, vid = res[res.subset == "all"], res[res.subset == "with_video"]
    md = [f"# Text + audio + video fusion ({split}, n={int(allv['n'].max())} responses; "
          f"{int(vid['n'].max())} with video)", "",
          "Spearman and CCC with 95% participant-bootstrap CIs. d = method minus reference on paired "
          "participant resamples (an interval containing 0 means no detectable difference). "
          "`weighted_ta` is the text + audio model recommended in v2. partial | len = Spearman after "
          "removing log word count from label and prediction.", "",
          "## Branch similarity (train out-of-fold, answers with video)", "",
          "| " + " | ".join(comp.columns) + " |", "|" + "---|" * len(comp.columns),
          *("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else
                              ("" if isinstance(v, float) else str(v)) for v in row) + " |"
            for row in comp.itertuples(index=False)), ""]
    for t in TARGETS:
        refs = ["text", "weighted_ta"] if t in SHARED else ["text"]
        cols = [f"d {m} vs {ref}" for ref in refs for m in ("Spearman", "CCC")]
        md += [f"## {t} (all val)", "",
               "| method | Spearman | CCC | RMSE | partial \\| len | " + " | ".join(cols) + " |",
               "|" + "---|" * (5 + len(cols))]
        for r in allv[allv.target == t].to_dict("records"):
            ds = [f"{r[f'd_{m}_vs_{ref}']:+.3f} {F.fmt_ci(r[f'd_{m}_vs_{ref}_ci95'])}"
                  if f"d_{m}_vs_{ref}" in r and pd.notna(r[f"d_{m}_vs_{ref}"]) else ""
                  for ref in refs for m in ("spearman", "ccc")]
            md.append(f"| {r['method']} | {r['spearman']:.3f} {F.fmt_ci(r['spearman_ci95'])} | {r['ccc']:.3f} "
                      f"{F.fmt_ci(r['ccc_ci95'])} | {r['rmse']:.3f} | {r['partial_spearman_len']:.3f} | "
                      + " | ".join(ds) + " |")
        ws = {k.split("/")[0]: v for k, v in chosen.items() if k.startswith("weighted_") and k.endswith(f"/{t}")}
        if ws:
            md += ["", "Weights (chosen on train OOF): " + "; ".join(
                f"{k}: " + ", ".join(f"{b[2:]} {x:.2f}" for b, x in v.items() if b.startswith("w_"))
                for k, v in ws.items()) + "."]
        md.append("")
    md += ["## Answers with video only", "",
           "| target | method | Spearman | CCC | d Spearman vs weighted_ta |", "|---|---|---|---|---|"]
    for r in vid.to_dict("records"):
        d = (f"{r['d_spearman_vs_weighted_ta']:+.3f} {F.fmt_ci(r['d_spearman_vs_weighted_ta_ci95'])}"
             if "d_spearman_vs_weighted_ta" in r and pd.notna(r.get("d_spearman_vs_weighted_ta")) else "")
        md.append(f"| {r['target']} | {r['method']} | {r['spearman']:.3f} | {r['ccc']:.3f} | {d} |")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="version name, e.g. v3_video; never overwritten")
    ap.add_argument("--eval-test", action="store_true",
                    help="also report test metrics; run once, after choosing on validation")
    args = ap.parse_args()
    out = F.OUT_DIR / args.tag
    pdir, rdir = out / "predictions", out / "reports"
    existing = out.exists()
    if existing and not args.eval_test:
        raise SystemExit(f"{out} already exists; pick a new --tag (versions are never overwritten)")
    if existing and (rdir / "fusion_test.csv").exists():
        raise SystemExit(f"{rdir / 'fusion_test.csv'} already exists; test is evaluated once per version")

    df, blocks = load_inputs()
    tr = (df["split"] == "train").to_numpy()
    folds = list(GroupKFold(n_splits=F.N_FOLDS, shuffle=True, random_state=SEED)
                 .split(np.zeros(tr.sum()), groups=df.loc[tr, "user_no"].to_numpy()))
    for c in [f"text_{t}" for t in TARGETS] + [f"audio_{t}" for t in SHARED]:
        assert df.loc[tr | (df.split == "val").to_numpy(), c].notna().all(), f"missing {c} predictions"

    late, chosen = late_fusion(df, tr, folds)
    early, chosen_early = early_fusion(df, blocks, tr, folds)
    methods, chosen = {**late, **early}, {**chosen, **chosen_early}

    comp = complementarity(df, tr)
    print(comp.round(3).to_string(index=False))
    if existing:
        splits = ["test"]
    else:
        pdir.mkdir(parents=True)
        rdir.mkdir(parents=True)
        for name, preds in methods.items():
            preds.to_parquet(pdir / f"{name}.parquet", index=False)
        (rdir / "fusion_config.json").write_text(json.dumps(
            {"tag": args.tag, "text_model": F.TEXT_MODEL, "video": "run_video_branch.py (robust Ridge)",
             "folds": "GroupKFold(5, shuffle, seed 42) by user_no on train",
             "n_pca": F.N_PCA, "simplex_step": STEP, "chosen": chosen}, indent=2, default=float),
            encoding="utf-8")
        for src in (Path(__file__), FUSION / "fuse.py", FUSION / "run_video_branch.py"):
            shutil.copyfile(src, out / src.name)
        splits = ["val"] + (["test"] if args.eval_test else [])
    for s in splits:
        res = pd.concat([evaluate(df, methods, s, False), evaluate(df, methods, s, True)])
        res.to_csv(rdir / f"fusion_{s}.csv", index=False)
        write_markdown(res, comp, chosen, s, rdir / f"fusion_{s}.md")
        print(res[res.subset == "all"][["method", "target", "spearman", "ccc", "rmse", "partial_spearman_len"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
