"""Stage 7, step 0: how reliable is each MIT Turker label?

The same 9 Mechanical Turk workers rated all 138 interviews (plus an EM-aggregated "AGGR" row). A label can only
validate a delivery category if raters agree on it, and a category's correlation must be read against that label's
ceiling. Per label:
  ICC(2,1)   agreement of a single rater (two-way random, absolute agreement)
  ICC(2,k)   reliability of the mean of the k raters  (= the ceiling for correlations with the rater mean)
  split-half Spearman-Brown corrected correlation of two random halves of the raters (median over 200 splits)
  rho_AGGR   Spearman of the AGGR label with the plain rater mean
Interviews with a missing rating are dropped per label (complete raters x interviews matrix).

Output  stage7_mit_validation/tables/label_reliability.csv (+ printed table; the Stage 7 report includes it)
Run:    python scripts/explainability/mit_label_reliability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402

LABELS = ["Overall", "RecommendHiring", "Colleague", "Engaged", "Excited", "EyeContact", "Smiled", "SpeakingRate",
          "NoFillers", "Friendly", "Paused", "EngagingTone", "StructuredAnswers", "Calm", "NotStressed", "Focused",
          "Authentic", "NotAwkward"]


def load_scores() -> pd.DataFrame:
    s = pd.read_csv(P.MIT_SCORES)
    s["Participant"] = s["Participant"].str.strip().str.upper()
    return s


def icc2(m: np.ndarray) -> tuple[float, float]:
    """m: interviews x raters, complete. Returns ICC(2,1), ICC(2,k) (Shrout & Fleiss)."""
    n, k = m.shape
    grand = m.mean()
    msr = k * ((m.mean(1) - grand) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - grand) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + grand) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    single = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    average = (msr - mse) / (msr + (msc - mse) / n)
    return float(single), float(average)


def split_half(m: np.ndarray, rng: np.random.Generator, n_splits: int = 200) -> float:
    k = m.shape[1]
    vals = []
    for _ in range(n_splits):
        perm = rng.permutation(k)
        a, b = m[:, perm[: k // 2]].mean(axis=1), m[:, perm[k // 2:]].mean(axis=1)
        r = spearmanr(a, b)[0]
        vals.append(2 * r / (1 + r))
    return float(np.median(vals))


def reliability() -> pd.DataFrame:
    s = load_scores()
    raters = s[s["Worker"] != "AGGR"]
    aggr = s[s["Worker"] == "AGGR"].set_index("Participant")
    rng = np.random.default_rng(0)
    rows = []
    for lab in LABELS:
        wide = raters.pivot_table(index="Participant", columns="Worker", values=lab, aggfunc="mean")
        wide = wide.loc[:, wide.notna().mean() > 0.9].dropna()
        m = wide.to_numpy(float)
        single, average = icc2(m)
        mean = wide.mean(axis=1)
        rows.append({"label": lab, "n_interviews": len(wide), "n_raters": m.shape[1], "icc2_1": single,
                     "icc2_k": average, "split_half_sb": split_half(m, rng),
                     "rho_aggr_vs_mean": spearmanr(mean, aggr.loc[wide.index, lab])[0],
                     "mean": float(m.mean()), "sd_of_interview_means": float(mean.std())})
    return pd.DataFrame(rows).sort_values("icc2_k", ascending=False)


def main() -> None:
    P.S7_TABLES.mkdir(parents=True, exist_ok=True)
    r = reliability()
    r.to_csv(P.S7_RELIABILITY, index=False)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
