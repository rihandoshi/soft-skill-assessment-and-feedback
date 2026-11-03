"""Model comparison beyond the summary table, plus the length decomposition.

1. Length decomposition (train split): is the "longer answers score lower" effect about word
   count, clip duration, or speaking rate? Rank (Spearman) and partial rank correlations.
2. For every saved model on --split (val by default):
     spearman, ccc               as in summarize_results.py
     ccc_recal                   CCC after a linear rescaling a + b*pred fitted on the model's
                                 out-of-fold TRAIN predictions (Ridge shrinks towards the mean)
     within_q_spearman           median Spearman inside each question (no credit for the question)
     partial_spearman_len        Spearman of pred with the label after both are adjusted for log
                                 word count: does the model carry signal beyond length?
     d_spearman_vs_len, d_ccc_vs_len   paired participant-bootstrap difference vs length_only
3. ens_linguistic_deberta: average of the linguistic Ridge and the newest DeBERTa predictions.

Writes outputs/text/reports/analysis_<split>.{csv,md} and length_decomposition.md.
Only --split val is meant for model selection.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from text_common import (OUT_DIR, SEED, TARGETS, cluster_bootstrap_ci, paired_bootstrap_diff,
                         regression_metrics, within_group_spearman)

N_BOOT = 2000


def rank_partial(y, x, ctrl) -> float:
    """Spearman partial correlation of y and x controlling for the columns in ctrl."""
    R = lambda v: stats.rankdata(v)
    C = np.column_stack([np.ones(len(y))] + [R(c) for c in ctrl])
    res = lambda v: R(v) - C @ np.linalg.lstsq(C, R(v), rcond=None)[0]
    return float(np.corrcoef(res(y), res(x))[0, 1])


def partial_ci(y, p, length, groups, n_boot=N_BOOT, seed=SEED):
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by_g[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        vals.append(rank_partial(y[idx], p[idx], [length[idx]]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def length_decomposition(df: pd.DataFrame) -> str:
    t = df[df["split"] == "train"].copy()
    t["words_per_s"] = t["n_words"] / t["clip_seconds"]
    lines = ["# Length decomposition (train, n=%d)" % len(t), "",
             "Spearman between the label and each measure; `a | b` = partial Spearman of the label "
             "with a, holding b fixed. words_per_s = transcript words / clip seconds.", "",
             f"Correlation of word count with clip seconds: {stats.spearmanr(t.n_words, t.clip_seconds)[0]:.2f}; "
             f"with words_per_s: {stats.spearmanr(t.n_words, t.words_per_s)[0]:.2f}.", "",
             "| target | words | seconds | words_per_s | words \\| seconds | seconds \\| words "
             "| words_per_s \\| seconds |", "|---|---|---|---|---|---|---|"]
    for y in TARGETS:
        sp = lambda c: stats.spearmanr(t[y], t[c])[0]
        lines.append(f"| {y} | {sp('n_words'):.2f} | {sp('clip_seconds'):.2f} | {sp('words_per_s'):.2f} | "
                     f"{rank_partial(t[y], t.n_words, [t.clip_seconds]):.2f} | "
                     f"{rank_partial(t[y], t.clip_seconds, [t.n_words]):.2f} | "
                     f"{rank_partial(t[y], t.words_per_s, [t.clip_seconds]):.2f} |")
    return "\n".join(lines) + "\n"


def recalibrate(pred_train, y_train, pred_eval):
    ok = np.isfinite(pred_train) & np.isfinite(y_train)
    if ok.sum() < 50 or np.std(pred_train[ok]) == 0:
        return None
    b, a = np.polyfit(pred_train[ok], y_train[ok], 1)
    return a + b * pred_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    feats = pd.read_parquet(OUT_DIR / "features" / "responses.parquet")
    rdir = OUT_DIR / "reports"
    if "clip_seconds" in feats:
        (rdir / "length_decomposition.md").write_text(length_decomposition(feats), encoding="utf-8")

    lab = feats[["id", "split", "question_id", "user_no", "log_n_words"] + TARGETS]
    preds = {f.stem: f for f in sorted((OUT_DIR / "predictions").glob("*.parquet"))}
    loaded = {name: pd.read_parquet(f)[["id"] + [f"pred_{t}" for t in TARGETS]] for name, f in preds.items()}
    deberta = sorted(n for n in loaded if n.startswith("finetune_"))
    if "linguistic" in loaded and deberta:
        newest = max(deberta, key=lambda n: preds[n].stat().st_mtime)
        ens = loaded["linguistic"].merge(loaded[newest], on="id", suffixes=("_a", "_b"))
        for t in TARGETS:
            ens[f"pred_{t}"] = (ens[f"pred_{t}_a"] + ens[f"pred_{t}_b"]) / 2
        loaded["ens_linguistic_deberta"] = ens[["id"] + [f"pred_{t}" for t in TARGETS]]
        print(f"ensemble uses linguistic + {newest}")

    ref = lab.merge(loaded["length_only"], on="id")
    rows = []
    for name, p in loaded.items():
        d = lab.merge(p, on="id")
        tr, ev = d[d["split"] == "train"], d[d["split"] == args.split]
        ref_ev = ref.set_index("id").loc[ev["id"]]
        for t in TARGETS:
            y, pr = ev[t].to_numpy(float), ev[f"pred_{t}"].to_numpy(float)
            g = ev["user_no"].to_numpy()
            r = {"model": name, "target": t, **{k: v for k, v in regression_metrics(y, pr).items()
                                                if k in ("spearman", "ccc", "r2")}}
            if np.allclose(pr, pr[0]):
                rows.append(r); continue
            rc = recalibrate(tr[f"pred_{t}"].to_numpy(float), tr[t].to_numpy(float), pr)
            r["ccc_recal"] = regression_metrics(y, rc)["ccc"] if rc is not None else np.nan
            r["within_q_spearman"] = within_group_spearman(y, pr, ev["question_id"])
            r["partial_spearman_len"] = rank_partial(y, pr, [ev["log_n_words"].to_numpy(float)])
            r["partial_len_ci95"] = partial_ci(y, pr, ev["log_n_words"].to_numpy(float), g)
            r["spearman_ci95"] = cluster_bootstrap_ci(y, pr, g, "spearman")
            if name != "length_only":
                p_ref = ref_ev[f"pred_{t}"].to_numpy(float)
                for m in ("spearman", "ccc"):
                    pt, lo, hi = paired_bootstrap_diff(y, pr, p_ref, g, m, N_BOOT)
                    r[f"d_{m}_vs_len"], r[f"d_{m}_vs_len_ci95"] = pt, (lo, hi)
            rows.append(r)
        print(f"done: {name}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(rdir / f"analysis_{args.split}.csv", index=False)

    f2 = lambda v: "" if not isinstance(v, (float, np.floating)) or np.isnan(v) else f"{v:.3f}"
    ci = lambda v: "" if not isinstance(v, tuple) else f"[{v[0]:+.2f}, {v[1]:+.2f}]"
    md = [f"# Text model analysis ({args.split}, n={int((lab['split'] == args.split).sum())} responses)", "",
          "d_* = model minus length_only on paired participant resamples (95% CI); an interval "
          "containing 0 means indistinguishable from word count. partial | len = Spearman with the "
          "label after adjusting both for log word count. ccc_recal = CCC after linear rescaling "
          "fitted on the model's out-of-fold train predictions.", ""]
    for t in TARGETS:
        md += [f"## {t}", "", "| model | Spearman | d Spearman vs len | CCC | CCC recal | d CCC vs len "
               "| within-question Spearman | partial \\| len |", "|---|---|---|---|---|---|---|---|"]
        for _, r in res[res["target"] == t].iterrows():
            md.append(f"| {r['model']} | {f2(r['spearman'])} | {f2(r.get('d_spearman_vs_len'))} "
                      f"{ci(r.get('d_spearman_vs_len_ci95'))} | {f2(r['ccc'])} | {f2(r.get('ccc_recal'))} | "
                      f"{f2(r.get('d_ccc_vs_len'))} {ci(r.get('d_ccc_vs_len_ci95'))} | "
                      f"{f2(r.get('within_q_spearman'))} | {f2(r.get('partial_spearman_len'))} "
                      f"{ci(r.get('partial_len_ci95'))} |")
        md.append("")
    (rdir / f"analysis_{args.split}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {rdir / f'analysis_{args.split}.md'}")


if __name__ == "__main__":
    main()
