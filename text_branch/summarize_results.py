"""Collect every text model's saved predictions into one comparison table.

Reads outputs/text/predictions/*.parquet and writes outputs/text/reports/summary_<split>.csv
and summary_<split>.md. Only --split val is meant for model selection; use --split test
once, after the final model has been chosen.
"""
from __future__ import annotations

import argparse

import pandas as pd

from text_common import OUT_DIR, TARGETS, cluster_bootstrap_ci, regression_metrics

ORDER = ["mean", "question_mean", "length_only", "duration_only", "linguistic", "embedding",
         "embedding_ling", "finetune_deberta-v3-base", "finetune_deberta-v3-base_v2", "linguistic_ridge", "linguistic_svr", "delivery_ridge",
         "delivery_svr", "linguistic_delivery_ridge", "linguistic_delivery_svr"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    args = ap.parse_args()

    labels = pd.read_parquet(OUT_DIR / "features" / "responses.parquet")[["id"] + TARGETS]
    rows = []
    for f in sorted((OUT_DIR / "predictions").glob("*.parquet")):
        d = pd.read_parquet(f)
        d = d[d["split"] == args.split].merge(labels, on="id")
        for t in TARGETS:
            y, p = d[t].to_numpy(float), d[f"pred_{t}"].to_numpy(float)
            r = {"model": f.stem, "target": t, **regression_metrics(y, p)}
            r["spearman_lo"], r["spearman_hi"] = cluster_bootstrap_ci(y, p, d["user_no"], "spearman")
            r["ccc_lo"], r["ccc_hi"] = cluster_bootstrap_ci(y, p, d["user_no"], "ccc")
            rows.append(r)
    res = pd.DataFrame(rows)
    res["model"] = pd.Categorical(res["model"], [m for m in ORDER if m in set(res["model"])]
                                  + sorted(set(res["model"]) - set(ORDER)))
    res = res.sort_values(["target", "model"])
    rdir = OUT_DIR / "reports"
    res.to_csv(rdir / f"summary_{args.split}.csv", index=False)

    lines = [f"# Text branch results ({args.split}, n={int(res['n'].iloc[0])} responses)", "",
             "Spearman / CCC with 95% participant-bootstrap CI; R2; RMSE.", ""]
    for t in TARGETS:
        lines += [f"## {t}", "", "| model | Spearman [95% CI] | CCC [95% CI] | R2 | RMSE |",
                  "|---|---|---|---|---|"]
        for _, r in res[res["target"] == t].iterrows():
            sp = "n/a" if pd.isna(r["spearman"]) else \
                f"{r['spearman']:.3f} [{r['spearman_lo']:.2f}, {r['spearman_hi']:.2f}]"
            cc = "n/a" if pd.isna(r["ccc_lo"]) else f"{r['ccc']:.3f} [{r['ccc_lo']:.2f}, {r['ccc_hi']:.2f}]"
            lines.append(f"| {r['model']} | {sp} | {cc} | {r['r2']:.3f} | {r['rmse']:.3f} |")
        lines.append("")
    (rdir / f"summary_{args.split}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
