"""diagnostics.py - audit how the data is structured, and what limits any model.

No training. Prints (and saves to results_v2/diagnostics.txt):

RecruitView2
  * videos per person, and which feature families exist (is gaze present?)
  * how strongly the target labels correlate with each other (are they really separate skills?)
  * between-person share of label variance (ICC)
  * how much label variance question_id / duration / video_quality explain (adjusted eta^2)
  * correlation of labels with video length / sample count / face rate
  * what the quality filter removes, and whether it removes low- or high-scoring videos
  * label means by video-length bucket (is the length effect monotone?)
  * detector check: what share of sampled frames really contain a detected face
MIT
  * RATER RELIABILITY per label from the per-worker ratings (ICC), i.e. the ceiling on how
    well ANY model can predict that label. The paper treated inter-rater agreement as a core step.
  * pre vs post intervention label shifts (each person is interviewed twice)
  * label-label correlations
  * within-person analysis: do changes in behaviour (post - pre) track changes in ratings?

Run:  python diagnostics.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import train as T
from aggregate import valid_mask

OUT = Path("results_v2")
LINES = []


def log(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    LINES.append(s)


def show(df, **kw):
    log(df.to_string(float_format=lambda v: f"{v:7.3f}", **kw))


# ----------------------------- statistics helpers ----------------------------
def icc1(values, groups):
    """One-way random-effects ICC. Returns (single-rater ICC, reliability of the group mean,
    effective group size). For RecruitView 'groups' = people, so ICC = between-person share."""
    d = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(groups)}).dropna()
    g = d.groupby("g").v
    n_i, N, a = g.size(), len(d), d.g.nunique()
    if a < 2 or N <= a:
        return np.nan, np.nan, np.nan
    grand = d.v.mean()
    msb = (n_i * (g.mean() - grand) ** 2).sum() / (a - 1)
    msw = ((d.v - g.transform("mean")) ** 2).sum() / (N - a)
    k0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    single = (msb - msw) / (msb + (k0 - 1) * msw)
    mean_k = (msb - msw) / msb if msb > 0 else np.nan
    return single, mean_k, k0


def adj_eta_sq(values, cats):
    """Adjusted eta^2 (epsilon^2): share of variance explained by a categorical, corrected
    for the number of levels (many levels inflate raw eta^2 by chance)."""
    d = pd.DataFrame({"v": np.asarray(values, float), "c": np.asarray(cats).astype(str)}).dropna()
    g = d.groupby("c").v
    K, N = d.c.nunique(), len(d)
    if K < 2 or N <= K:
        return np.nan
    grand = d.v.mean()
    ssb = (g.size() * (g.mean() - grand) ** 2).sum()
    sst = ((d.v - grand) ** 2).sum()
    msw = ((d.v - g.transform("mean")) ** 2).sum() / (N - K)
    return (ssb - (K - 1) * msw) / sst


def feature_inventory(df):
    fam = {"AU": [], "emotion": [], "pose": [], "valence/arousal": [], "gaze": [], "smile/nod/shake": [],
           "face distances": [], "ratios": [], "other": []}
    emo = ("anger", "disgust", "fear", "happy", "happiness", "sad", "sadness", "surprise",
           "neutral", "contempt")
    for c in df.attrs["feature_cols"]:
        l = c.lower()
        if re.match(r"au\d+", l): fam["AU"].append(c)
        elif l.startswith(emo): fam["emotion"].append(c)
        elif l.startswith(("pitch", "yaw", "roll")): fam["pose"].append(c)
        elif l.startswith(("valence", "arousal")): fam["valence/arousal"].append(c)
        elif "gaze" in l: fam["gaze"].append(c)
        elif l.startswith(("smile_intensity", "head_nod", "head_shake")): fam["smile/nod/shake"].append(c)
        elif l.endswith("_ratio") or "_ratio_" in l: fam["ratios"].append(c)
        elif l.startswith(("inbr", "otbr", "eyeo", "olip", "ilip", "lipc", "brow_height")): fam["face distances"].append(c)
        else: fam["other"].append(c)
    log("Feature families (count):", {k: len(v) for k, v in fam.items() if v})
    if not fam["gaze"]:
        log("  NOTE: no gaze features in this feature table (gaze is the natural eye-contact signal).")
    if fam["other"]:
        log("  other:", fam["other"][:12])


# ----------------------------- RecruitView -----------------------------------
def audit_recruitview():
    log("\n" + "=" * 70 + "\nRECRUITVIEW2\n" + "=" * 70)
    old = (T.MIN_SAMPLES, T.MIN_FACE_RATE)
    T.MIN_SAMPLES, T.MIN_FACE_RATE = 0, 0                 # load everything to see what the filter drops
    full = T.load_recruitview()
    T.MIN_SAMPLES, T.MIN_FACE_RATE = old
    feature_inventory(full)

    per_user = full.groupby("user_no").size()
    log(f"\nVideos per person: mean {per_user.mean():.1f}, min {per_user.min()}, max {per_user.max()} "
        f"({len(per_user)} people)")

    targets = list(T.RV_TASKS.values())
    others = [c for c in ("overall_performance", "interview_score", "answer_score", "speaking_skills",
                          "openness", "conscientiousness", "agreeableness", "neuroticism")
              if c in full.columns]
    cols = [c for c in targets + others if c in full.columns]
    log("\nSpearman correlation between labels (are the targets really separate skills?):")
    show(full[cols].corr(method="spearman"))

    log("\nBetween-person share of label variance (ICC; 0 = person irrelevant, 1 = all variance is person-level):")
    log(pd.Series({c: icc1(full[c], full["user_no"])[0] for c in targets}).round(3).to_string())

    log("\nVariance explained by metadata (adjusted eta^2; ~0 = nothing):")
    rows = {}
    for t in targets:
        r = {}
        for cat in ("question_id", "duration", "video_quality"):
            if cat in full.columns:
                r[cat] = adj_eta_sq(full[t], full[cat])
        rows[t] = r
    show(pd.DataFrame(rows).T)

    log("\nSpearman of labels with video length / samples / face rate:")
    nuis = [c for c in ("duration_sec", "n_samples", "face_rate") if c in full.columns]
    show(pd.DataFrame({t: {n: spearmanr(full[t], full[n], nan_policy="omit")[0] for n in nuis}
                       for t in targets}))

    if "duration_sec" in full.columns:
        bins = pd.cut(full.duration_sec, [0, 5, 10, 15, 30, 45, 60, 100000],
                      labels=["<=5s", "5-10s", "10-15s", "15-30s", "30-45s", "45-60s", ">60s"])
        tab = full.groupby(bins, observed=True)[targets].mean()
        tab["n_videos"] = full.groupby(bins, observed=True).size()
        log("\nMean label by video length (z-scored labels; 0 = average video):")
        show(tab)

    cache = Path("frame_features/recruitview")
    files = sorted(cache.glob("*.csv"))[:300] if cache.exists() else []
    if files:
        tot = ok = low = 0
        for f in files:
            fr = pd.read_csv(f)
            vm = valid_mask(fr)
            tot, ok, low = tot + len(fr), ok + int(vm.sum()), low + int(vm.mean() < 0.5)
        log(f"\nDetector check on {len(files)} cached videos: {ok}/{tot} sampled frames ({ok / tot:.1%}) "
            f"contain a detected face; {low} videos have <50% detected.")
        if ok < tot:
            log("  => Py-Feat returns all-NaN rows for frames without a face. The old n_samples / face_rate "
                "counted those rows; re-run extract_features_recruitview.py (uses the cache) to fix.")

    if {"n_samples", "face_rate"}.issubset(full.columns):
        kept = (full.n_samples >= T.MIN_SAMPLES) & (full.face_rate >= T.MIN_FACE_RATE)
        log(f"\nQuality filter (n_samples>={T.MIN_SAMPLES}, face_rate>={T.MIN_FACE_RATE}) drops "
            f"{(~kept).sum()} of {len(full)} videos. Mean label (kept vs dropped):")
        show(pd.DataFrame({t: {"kept": full.loc[kept, t].mean(), "dropped": full.loc[~kept, t].mean()}
                           for t in targets}).T)


# ----------------------------- MIT -------------------------------------------
def audit_mit():
    log("\n" + "=" * 70 + "\nMIT INTERVIEW DATASET\n" + "=" * 70)
    sc = pd.read_csv(T.MIT_SCORES)
    wcol, pcol = T.MIT_WORKER_COL, T.MIT_ID_COL
    is_aggr = sc[wcol].astype(str).str.upper() == "AGGR"
    raters, aggr = sc[~is_aggr], sc[is_aggr]
    rating_cols = [c for c in sc.select_dtypes("number").columns if c.lower() != "total"]
    per_video = raters.groupby(pcol).size()
    log(f"Per-rater rows: {len(raters)} over {raters[pcol].nunique()} interviews "
        f"({per_video.mean():.1f} raters each)")

    log("\nRATER RELIABILITY per label (ICC). 'mean-of-raters' is the reliability of the averaged score,\n"
        "which is what we predict; its square root is a rough ceiling on achievable correlation r.")
    rel = {}
    for c in rating_cols:
        single, mean_k, _ = icc1(raters[c], raters[pcol].str.lower())
        rel[c] = {"single_rater_ICC": single, "mean_of_raters_ICC": mean_k,
                  "ceiling_r": np.sqrt(max(mean_k, 0)) if np.isfinite(mean_k) else np.nan,
                  "modelled": "yes" if c in T.MIT_TASKS.values() else ""}
    rel = pd.DataFrame(rel).T
    for c in ("single_rater_ICC", "mean_of_raters_ICC", "ceiling_r"):
        rel[c] = pd.to_numeric(rel[c])
    show(rel.sort_values("mean_of_raters_ICC", ascending=False))

    a = aggr.copy()
    a["pid"] = a[pcol].astype(str).str.lower()
    a["phase"] = np.where(a.pid.str.startswith("pp"), "post", "pre")
    a["num"] = a.pid.str.extract(r"(\d+)")[0].astype(int)
    labs = [c for c in T.MIT_TASKS.values() if c in a.columns]

    log("\nPre vs post intervention (aggregated ratings, paired by person; Wilcoxon p):")
    wide = a.pivot(index="num", columns="phase", values=labs).dropna()
    out = {}
    for c in labs:
        pre, post = wide[(c, "pre")], wide[(c, "post")]
        try:
            p = wilcoxon(post, pre).pvalue
        except ValueError:
            p = np.nan
        out[c] = {"mean_pre": pre.mean(), "mean_post": post.mean(), "diff": (post - pre).mean(), "p": p}
    show(pd.DataFrame(out).T)

    log("\nSpearman correlation between the modelled labels:")
    show(a[labs].corr(method="spearman"))

    m = T.load_mit()
    feature_inventory(m)
    feats = [c for c in m.attrs["feature_cols"] if pd.api.types.is_numeric_dtype(m[c])]
    m["num"] = m["participant"].astype(int)
    pre = m[m.phase == "pre"].set_index("num")
    post = m[m.phase == "post"].set_index("num")
    both = pre.index.intersection(post.index)
    log(f"\nWITHIN-PERSON analysis (post - pre) over {len(both)} people: do changes in behaviour track "
        "changes in ratings?\n(top 5 features by |Spearman| per label; n is small, treat as exploratory)")
    for c in labs:
        if c not in m.columns:
            continue
        dy = post.loc[both, c] - pre.loc[both, c]
        res = []
        for f in feats:
            dx = post.loc[both, f] - pre.loc[both, f]
            ok = dx.notna() & dy.notna()
            if ok.sum() > 10 and dx[ok].nunique() > 2:
                r, p = spearmanr(dx[ok], dy[ok])
                res.append((f, r, p))
        res = sorted(res, key=lambda t: -abs(t[1]))[:5]
        log(f"  {c}: " + ", ".join(f"{f}({r:+.2f}, p={p:.3f})" for f, r, p in res))


def main():
    if T.RECRUITVIEW_FEATURES.exists() and T.RECRUITVIEW_META.exists():
        audit_recruitview()
    if T.MIT_FEATURES.exists() and T.MIT_SCORES.exists():
        audit_mit()
    OUT.mkdir(exist_ok=True)
    (OUT / "diagnostics.txt").write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nSaved {OUT / 'diagnostics.txt'}")


if __name__ == "__main__":
    main()
