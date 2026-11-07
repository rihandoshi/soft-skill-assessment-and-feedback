"""
evaluate_metrics.py  -  compute every evaluation metric from a predictions file and write a Markdown report.

Works on any predictions CSV that has an `id` column (RecruitView metadata id) and one or more
`pred_<something>` columns, e.g. results_v3/preds_v3_pm.csv from train_v3.py. Teammates can use it for
text/voice predictions too, as long as the file follows the same layout (id + pred_* columns, all rows or a subset).

Usage (project folder, venv active):
    python evaluate_metrics.py                                   # evaluates results_v3/preds_v3_pm.csv and preds_v3_nopm.csv
    python evaluate_metrics.py --preds my_preds.csv --out report.md
    python evaluate_metrics.py --preds a.csv b.csv --pred-col pred_composite

Metrics per (prediction column x target x split), split = train_oof / val / test:
  Correlation : Pearson r (+95% CI, bootstrap over PEOPLE), Spearman rho
  Length      : partial Pearson controlling for log(video length) ("beyond length"),
                and the log-length-only baseline on the same rows
  Person      : within-person Pearson (pred and label centred on each person's mean; people with >=2 answers),
                person-level Pearson (means per person)
  Error       : R2, RMSE, MAE after a linear calibration pred -> label fitted on TRAIN rows only
                (predictions from ensembles are on a z-score scale, so raw R2 would be meaningless)
  Ranking     : AUC for bottom third and top third (cut-offs = TRAIN label tertiles)
  Classes     : 3-class accuracy and balanced accuracy (tertiles of TRAIN labels and of TRAIN-calibrated
                predictions), with the majority-class baseline
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
META = ROOT / "RecruitView2" / "metadata.jsonl"
SPLIT_DIR = ROOT / "RecruitView2" / "Split Dataset"
FEATS = ROOT / "features_recruitview.csv"  # only used for duration_sec

TARGETS = ["confidence_score", "speaking_skills", "answer_score", "facial_expression",
           "overall_performance", "extraversion", "interview_score"]
SPLITS = ["train_oof", "val", "test"]
N_BOOT = 1000
SEED = 0


# ------------------------------------------------------------------ data
def load_meta():
    meta = pd.read_json(META, lines=True)
    meta["id"] = meta["id"].astype(int)
    meta["user_no"] = meta.user_no.astype(int)
    meta["file_name"] = meta.file_name.astype(str).str.split("/").str[-1]
    split = {}
    for s in ["train", "val", "test"]:
        for u in (SPLIT_DIR / f"{s}_users.txt").read_text().split():
            split[int(u)] = s
    meta["split"] = meta.user_no.map(split).replace({"train": "train_oof"})
    if FEATS.exists():
        f = pd.read_csv(FEATS, usecols=["file_name", "duration_sec"])
        f["file_name"] = f.file_name.astype(str).str.split("/").str[-1]
        meta = meta.merge(f, on="file_name", how="left")
    else:
        meta["duration_sec"] = np.nan
    meta["log_dur"] = np.log1p(meta.duration_sec)
    return meta


# ------------------------------------------------------------------ metric helpers
def safe_r(a, b, fn=pearsonr):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return fn(a, b)[0]


def boot_ci(p, y, g, rng):
    users = np.unique(g)
    by_u = {u: np.where(g == u)[0] for u in users}
    rs = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by_u[u] for u in rng.choice(users, len(users))])
        rs.append(safe_r(p[idx], y[idx]))
    return np.nanpercentile(rs, [2.5, 97.5])


def partial_r(p, y, ctrl):
    ok = np.isfinite(ctrl)
    if ok.sum() < 5:
        return np.nan
    X = np.c_[np.ones(ok.sum()), ctrl[ok]]
    res = lambda a: a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    return safe_r(res(p[ok]), res(y[ok]))


def within_r(p, y, g):
    d = pd.DataFrame(dict(p=p, y=y, g=g))
    d = d[d.groupby("g").g.transform("size") >= 2]
    if len(d) < 5:
        return np.nan
    return safe_r((d.p - d.groupby("g").p.transform("mean")).values,
                  (d.y - d.groupby("g").y.transform("mean")).values)


def person_r(p, y, g):
    d = pd.DataFrame(dict(p=p, y=y, g=g)).groupby("g").mean()
    return safe_r(d.p.values, d.y.values)


def auc(y_bin, p):
    return roc_auc_score(y_bin, p) if 0 < y_bin.sum() < len(y_bin) else np.nan


def tertile(x, cuts):
    return np.digitize(x, cuts)


# ------------------------------------------------------------------ evaluation
def evaluate(df, pred_col, target, rng):
    """df has columns split, user_no, log_dur, target, pred_col."""
    d = df[df[pred_col].notna() & df[target].notna()]
    trn = d[d.split == "train_oof"]
    rows = []
    # calibration + tertile cut-offs from TRAIN rows only
    if len(trn) >= 10:
        a, b = np.polyfit(trn[pred_col], trn[target], 1)
        y_cuts = np.percentile(trn[target], [100 / 3, 200 / 3])
        p_cuts = np.percentile(a * trn[pred_col] + b, [100 / 3, 200 / 3])
        a_len, b_len = np.polyfit(trn.log_dur.fillna(trn.log_dur.median()), trn[target], 1) \
            if trn.log_dur.notna().any() else (np.nan, np.nan)
    else:
        a = b = np.nan
        y_cuts = p_cuts = None
    for s in SPLITS:
        x = d[d.split == s]
        if len(x) < 5:
            continue
        p, y, g, L = x[pred_col].values, x[target].values, x.user_no.values, x.log_dur.values
        r = dict(pred=pred_col, target=target, split=s, n=len(x), n_people=len(np.unique(g)))
        r["pearson"] = safe_r(p, y)
        r["ci_lo"], r["ci_hi"] = boot_ci(p, y, g, rng)
        r["spearman"] = safe_r(p, y, spearmanr)
        r["partial_len"] = partial_r(p, y, L)
        r["len_only_pearson"] = safe_r(-L[np.isfinite(L)], y[np.isfinite(L)])
        r["within_person"] = within_r(p, y, g)
        r["person_level"] = person_r(p, y, g)
        if np.isfinite(a):
            pc = a * p + b
            r["r2"] = 1 - np.sum((y - pc) ** 2) / np.sum((y - y.mean()) ** 2)
            r["rmse"] = np.sqrt(np.mean((y - pc) ** 2))
            r["mae"] = np.mean(np.abs(y - pc))
            if np.isfinite(a_len):
                pl = a_len * np.nan_to_num(L, nan=np.nanmedian(L)) + b_len
                r["r2_len_only"] = 1 - np.sum((y - pl) ** 2) / np.sum((y - y.mean()) ** 2)
            yc, pcl = tertile(y, y_cuts), tertile(pc, p_cuts)
            r["auc_low"] = auc((yc == 0).astype(int), -p)
            r["auc_high"] = auc((yc == 2).astype(int), p)
            r["acc3"] = np.mean(yc == pcl)
            r["bacc3"] = balanced_accuracy_score(yc, pcl)
            r["acc3_majority"] = np.bincount(yc, minlength=3).max() / len(yc)
        rows.append(r)
    return rows


# ------------------------------------------------------------------ markdown
def fmt(v, nd=3):
    return "" if pd.isna(v) else f"{v:.{nd}f}"


def md_table(df, cols, headers):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            if c == "ci":
                cells.append(f"[{fmt(r.ci_lo, 2)}, {fmt(r.ci_hi, 2)}]")
            elif isinstance(r[c], str):
                cells.append(r[c])
            elif c in ("n", "n_people"):
                cells.append(str(int(r[c])))
            else:
                cells.append(fmt(r[c]))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_md(res, sources, out_path):
    L = [f"# Evaluation report", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} by `evaluate_metrics.py`.", "",
         "Predictions: " + ", ".join(f"`{s}`" for s in sources), "",
         "Protocol: team split by `user_no` (train/val/test user lists in `RecruitView2/Split Dataset/`). "
         "Train rows are out-of-fold predictions; calibration, tertile cut-offs and the length-only "
         "baseline are fitted on train rows only. CIs are 95% bootstrap over people "
         f"({N_BOOT} resamples).", "",
         "## How to read the columns", "",
         "- **Pearson / Spearman**: correlation of prediction with label.",
         "- **beyond length**: partial Pearson controlling for log(video length) - what the model adds beyond answer length.",
         "- **length only**: Pearson of -log(video length) alone on the same rows (the baseline to beat).",
         "- **within person**: does the model rank one person's answers correctly? (centred on each person's mean)",
         "- **person level**: correlation of per-person mean prediction vs mean label.",
         "- **R2 / RMSE / MAE**: after a linear calibration fitted on train; R2 len-only is the same for the length baseline.",
         "- **AUC low / high**: separating the bottom / top third of labels (0.5 = chance).",
         "- **acc3 / bacc3**: 3-class (tertile) accuracy and balanced accuracy; majority = always predicting the most common class.",
         ""]
    val, test = res[res.split == "val"], res[res.split == "test"]
    if len(val) and len(test):
        pick = val.loc[val.groupby("target").pearson.idxmax(), ["target", "source", "pred", "pearson"]]
        pick = pick.rename(columns={"pearson": "val_pearson"})
        best = pick.merge(test, on=["target", "source", "pred"])
        best["choice"] = best.source.str.replace(".csv", "", regex=False) + " / " + best.pred
        L += ["## Headline: column chosen on VAL, reported on TEST", "",
              md_table(best, ["target", "choice", "val_pearson", "pearson", "ci", "spearman", "partial_len",
                              "len_only_pearson", "within_person"],
                       ["target", "file / prediction", "VAL Pearson", "TEST Pearson", "95% CI", "Spearman",
                        "beyond length", "length only", "within person"]),
              "", "The choice is made on VAL only, so the TEST numbers are an honest estimate.", ""]
    for src in res.source.unique():
        L += [f"## `{src}`", ""]
        rs = res[res.source == src]
        for pc in rs.pred.unique():
            rp = rs[rs.pred == pc]
            L += [f"### `{pc}`", "", "**Correlation**", "",
                  md_table(rp, ["target", "split", "n", "n_people", "pearson", "ci", "spearman", "partial_len",
                                "len_only_pearson", "within_person", "person_level"],
                           ["target", "split", "n", "people", "Pearson", "95% CI", "Spearman", "beyond length",
                            "length only", "within person", "person level"]), ""]
            if "r2" in rp:
                L += ["**Error and classification** (calibrated on train)", "",
                      md_table(rp, ["target", "split", "r2", "r2_len_only", "rmse", "mae", "auc_low", "auc_high",
                                    "acc3", "bacc3", "acc3_majority"],
                               ["target", "split", "R2", "R2 len-only", "RMSE", "MAE", "AUC low", "AUC high",
                                "acc3", "bacc3", "majority"]), ""]
    out_path.write_text("\n".join(L), encoding="utf-8")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", default=[str(ROOT / "results_v3" / "preds_v3_pm.csv"),
                                                   str(ROOT / "results_v3" / "preds_v3_nopm.csv")])
    ap.add_argument("--pred-col", nargs="*", help="only these pred_* columns (default: all)")
    ap.add_argument("--targets", nargs="*", default=TARGETS)
    ap.add_argument("--all-pairs", action="store_true", help="evaluate every pred column against every target")
    ap.add_argument("--out", default=str(ROOT / "results_v3" / "evaluation_report.md"))
    args = ap.parse_args()

    meta = load_meta()
    rng = np.random.default_rng(SEED)
    all_rows = []
    sources = []
    for path in args.preds:
        path = Path(path)
        if not path.exists():
            print(f"skip (not found): {path}")
            continue
        p = pd.read_csv(path)
        p["id"] = p["id"].astype(int)
        pcols = args.pred_col or [c for c in p.columns if c.startswith("pred_")]
        df = meta.merge(p[["id"] + pcols], on="id", how="inner")
        print(f"{path.name}: {len(df)} rows matched, columns {pcols}")
        for pc in pcols:
            own = pc[len("pred_"):]
            # default pairing: pred_<target> -> that target only; any other column (e.g. pred_composite,
            # a fusion score) -> every target. --all-pairs evaluates every column against every target.
            tlist = args.targets if (args.all_pairs or own not in TARGETS) else [own]
            for t in tlist:
                if t not in args.targets:
                    continue
                for r in evaluate(df, pc, t, rng):
                    r["source"] = path.name
                    all_rows.append(r)
        sources.append(path.name)
    if not all_rows:
        raise SystemExit("nothing evaluated")
    res = pd.DataFrame(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out.with_suffix(".csv"), index=False)
    write_md(res, sources, out)
    print(f"wrote {out} and {out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
