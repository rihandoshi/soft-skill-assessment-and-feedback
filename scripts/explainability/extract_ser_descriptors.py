"""Stage 5, step 2: answer-level vocal confidence & composure measures from the SER windows.

Inputs   stage5_ser/tables/windows_{model}.csv (run_ser_windows.py), the active descriptor tables (keys, duplicates),
         speaker pitch settings + cached Praat frames (window pitch for `tension_rise`)
Outputs  stage5_ser/tables/segment_ser.csv      one row per answer, merged into the descriptor tables by
                                                descriptor_tables.py (so Stages 2-3 see the columns)
         stage5_ser/tables/windows_{model}.csv  rewritten with speaker-relative z-scores per window
         stage5_ser/descriptor_registry.json, stage5_ser/report.md

Why speaker-normalise. Raw SER values partly describe the voice, not the moment (Stage 3: the hand-made composure
proxies jitter / shimmer were 80-85% speaker identity). Every window is therefore also expressed as a z-score
against the SAME speaker's windows in their OTHER recordings (leave-one-out; duplicate copies of the same recording
excluded). For a new user, "other recordings" = the other answers of the same interview. No labels are used.

Answer-level columns (only windows with >= 50% VAD speech):
  ser_dominance_mean / ser_arousal_mean   mean over the answer (Stage 2 adds population percentiles + speaker
                                          self-delta, which is what the composure class uses)
  ser_valence_mean                        diagnostic (no audio category; positivity belongs to content)
  ser_dominance_drop_sd (+ start / end)   largest fall of the smoothed dominance curve within the answer, in units of
                                          the speaker's window SD ("composure_drop")
  ser_tension_run_s (+ start / end)       longest run of consecutive windows with arousal z > 1 AND pitch z > 1
                                          ("tension_rise": more activated AND higher-pitched than this speaker's usual)
  ser_dominance_low_start_s / _end_s      where the smoothed dominance is lowest (timestamp for `low_dominance`)

Run:  python scripts/explainability/extract_ser_descriptors.py [--model audeering]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import extract_delivery_descriptors as X  # noqa: E402

MIN_SPEECH_FRAC = 0.5
MIN_SPEAKER_WINDOWS = 10
MIN_VOICED_FRAMES_WIN = 50
SMOOTH = 3                 # windows in the moving mean (~8 s of audio)
Z_TENSION = 1.0
HOP_S = 2.0
ATTRS = ("arousal", "dominance", "valence")


# ============================================================================================
def window_pitch(windows: pd.DataFrame, clips: pd.DataFrame, settings: pd.DataFrame, workers: int) -> np.ndarray:
    """Median voiced F0 per window in semitones relative to the speaker's reference F0 (Stage 1 frames)."""
    jobs = [(r.id, r.audio_path, settings.loc[r.participant, "pitch_floor_hz"], settings.loc[r.participant, "pitch_ceiling_hz"])
            for r in clips.itertuples()]
    frames = X.run_parallel(X._pass2, jobs, workers, "pass2 (cached)")
    part = clips.set_index("id")["participant"]
    out = np.full(len(windows), np.nan)
    for cid, idx in windows.groupby("id").indices.items():
        if cid not in frames:
            continue
        t, f0 = frames[cid]["pitch_t"].astype(float), frames[cid]["f0"].astype(float)
        ref = settings.loc[part[cid], "reference_f0_hz"]
        for i in idx:
            sel = (t >= windows["t_start"].iat[i]) & (t < windows["t_end"].iat[i]) & (f0 > 0)
            if sel.sum() >= MIN_VOICED_FRAMES_WIN:
                out[i] = float(np.median(12 * np.log2(f0[sel] / ref)))
    return out


def speaker_z(windows: pd.DataFrame, keys: pd.DataFrame, column: str) -> np.ndarray:
    """z of each window vs the same speaker's windows in their other (non-duplicate) recordings."""
    w = windows.merge(keys[["id", "participant", "group"]], on="id", how="left")
    z = np.full(len(w), np.nan)
    for _, g in w.groupby("participant"):
        vals = g[column].to_numpy(float)
        grp = g["group"].to_numpy()
        ok = np.isfinite(vals) & (g["speech_frac"].to_numpy() >= MIN_SPEECH_FRAC)
        for gr in np.unique(grp):
            other = ok & (grp != gr)
            if other.sum() < MIN_SPEAKER_WINDOWS:
                continue
            mu, sd = vals[other].mean(), vals[other].std(ddof=1)
            if sd > 0:
                rows = np.flatnonzero(grp == gr)
                z[g.index[rows]] = (vals[rows] - mu) / sd
    return z


def longest_run(mask: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> tuple[float, float, float]:
    """Longest run of True over consecutive windows (consecutive = start times one hop apart)."""
    best, cur = (0.0, np.nan, np.nan), None
    for i in range(len(mask)):
        if mask[i]:
            if cur is not None and abs(t0[i] - t0[i - 1] - HOP_S) < 1e-6 and mask[i - 1]:
                cur = (cur[0], i)
            else:
                cur = (i, i)
            dur = t1[cur[1]] - t0[cur[0]]
            if dur > best[0]:
                best = (float(dur), float(t0[cur[0]]), float(t1[cur[1]]))
        else:
            cur = None
    return best


def answer_measures(g: pd.DataFrame) -> dict:
    s = g[g["speech_frac"] >= MIN_SPEECH_FRAC].sort_values("t_start")
    out = {"ser_n_windows": len(s)}
    for a in ATTRS:
        out[f"ser_{a}_mean"] = float(s[a].mean()) if len(s) else np.nan
    nan = {"ser_dominance_drop_sd": np.nan, "ser_drop_start_s": np.nan, "ser_drop_end_s": np.nan,
           "ser_tension_run_s": np.nan, "ser_tension_start_s": np.nan, "ser_tension_end_s": np.nan,
           "ser_dominance_low_start_s": np.nan, "ser_dominance_low_end_s": np.nan}
    out.update(nan)
    if not len(s):
        return out
    t0, t1 = s["t_start"].to_numpy(), s["t_end"].to_numpy()
    dz = s["dominance_z"].to_numpy()
    if np.isfinite(dz).sum() >= 1:
        smooth = pd.Series(dz).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy()
        lo = int(np.nanargmin(smooth))
        out["ser_dominance_low_start_s"], out["ser_dominance_low_end_s"] = float(t0[lo]), float(t1[lo])
        if len(smooth) >= SMOOTH + 1:
            peak_before = np.fmax.accumulate(np.where(np.isfinite(smooth), smooth, -np.inf))
            drop = peak_before - smooth
            j = int(np.nanargmax(np.where(np.isfinite(drop), drop, -np.inf)))
            i = int(np.flatnonzero(smooth[:j + 1] == peak_before[j])[0])
            out["ser_dominance_drop_sd"] = float(drop[j])
            out["ser_drop_start_s"], out["ser_drop_end_s"] = float(t0[i]), float(t1[j])
    az, fz = s["arousal_z"].to_numpy(), s["f0_z"].to_numpy()
    if np.isfinite(az).any() and np.isfinite(fz).any():
        run = longest_run((az > Z_TENSION) & (fz > Z_TENSION), t0, t1)
        out["ser_tension_run_s"], out["ser_tension_start_s"], out["ser_tension_end_s"] = run
    return out


# ============================================================================================
def registry(model: str) -> dict:
    E, D = X.E, X.D
    src = f"[Stage 5: {model} dimensional SER (MSP-Podcast), 4-s windows, 2-s hop, windows with >= 50% speech.]"
    reg = {}

    def add(name, category, status, unit, definition):
        reg[f"segment.{name}"] = {"level": "segment", "name": name, "category": category, "status": status,
                                  "unit": unit, "definition": f"{definition} {src}"}
    add("ser_dominance_mean", "composure", E, "0-1",
        "Mean predicted dominance (listeners' 'sounds assured / in control'). Compared with the speaker's own other "
        "answers (Stage 2 self-delta) for the composure class; the raw value partly describes the voice.")
    add("ser_arousal_mean", "composure", E, "0-1", "Mean predicted arousal (activation / energy; high + raised pitch = tension).")
    add("ser_valence_mean", "diagnostic", D, "0-1", "Mean predicted valence (positivity). No audio category; diagnostic.")
    add("ser_n_windows", "diagnostic", D, "count", "Windows with >= 50% speech.")
    add("ser_dominance_drop_sd", "composure", E, "speaker SD",
        "Largest fall of the smoothed (3-window) speaker-relative dominance curve within the answer.")
    add("ser_tension_run_s", "composure", E, "s",
        f"Longest run of consecutive windows with arousal z > {Z_TENSION} AND window pitch z > {Z_TENSION} (both vs the "
        "speaker's other recordings).")
    for n in ("ser_drop_start_s", "ser_drop_end_s", "ser_tension_start_s", "ser_tension_end_s",
              "ser_dominance_low_start_s", "ser_dominance_low_end_s"):
        add(n, "diagnostic", D, "s", "Timestamp for the evidence JSON.")
    return reg


def icc1(df: pd.DataFrame, col: str) -> float:
    d = df[["participant", col]].dropna()
    g = d.groupby("participant")[col]
    k = g.size()
    d = d[d["participant"].isin(k[k >= 2].index)]
    g = d.groupby("participant")[col]
    n, n_groups = len(d), g.ngroups
    if n_groups < 3:
        return np.nan
    grand = d[col].mean()
    msb = (g.size() * (g.mean() - grand) ** 2).sum() / (n_groups - 1)
    msw = ((d[col] - g.transform("mean")) ** 2).sum() / (n - n_groups)
    k0 = (n - (g.size() ** 2).sum() / n) / (n_groups - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def write_report(model: str, windows: pd.DataFrame, ser: pd.DataFrame, seg: pd.DataFrame) -> None:
    L = []
    add = L.append
    u = seg[~seg["is_duplicate"]].merge(ser, on="id")
    tr = u[u["split"] == "train"]
    add("# Stage 5: vocal confidence & composure (pretrained dimensional SER)\n")
    add(f"Generated by `scripts/explainability/extract_ser_descriptors.py` from `run_ser_windows.py` ({model}: "
        "`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, used as-is, no training). 4-s windows, 2-s hop. "
        "Statistics on unique recordings; correlations on train.\n")
    add("## 1. Coverage\n")
    sp = windows["speech_frac"] >= MIN_SPEECH_FRAC
    add(f"- Windows: {len(windows):,} ({sp.mean():.0%} with >= 50% speech) in {windows['id'].nunique()} clips; answers with "
        f">= 1 speech window: {(u['ser_n_windows'] > 0).mean():.1%}.")
    add(f"- Speaker-relative z available for {np.isfinite(windows.loc[sp, 'dominance_z']).mean():.0%} of speech windows "
        f"(needs >= {MIN_SPEAKER_WINDOWS} windows in the speaker's other recordings).\n")
    add("## 2. Does SER describe the moment or the person?\n")
    add("ICC(1) = share of answer-level variance that is between speakers (Stage 3 found 0.80-0.85 for jitter / shimmer, "
        "0.97 for absolute pitch). Window-level: correlation of adjacent windows (smoothness of the curve).\n")
    add("| Measure | ICC (speaker share) | adjacent-window Spearman |")
    add("|---|---|---|")
    for a in ATTRS:
        w = windows[sp].sort_values(["id", "t_start"])
        nxt = w.groupby("id")[a].shift(-1)
        ok = nxt.notna()
        add(f"| `ser_{a}_mean` | {icc1(u, f'ser_{a}_mean'):.2f} | {spearmanr(w.loc[ok, a], nxt[ok])[0]:.2f} |")
    ws = windows[sp]
    add(f"\n**Are the three outputs different measurements?** Spearman across speech windows: arousal-dominance "
        f"**{ws['arousal'].corr(ws['dominance'], 'spearman'):.2f}** (speaker-relative z: "
        f"{ws['arousal_z'].corr(ws['dominance_z'], 'spearman'):.2f}), arousal-valence "
        f"{ws['arousal'].corr(ws['valence'], 'spearman'):.2f}, dominance-valence {ws['dominance'].corr(ws['valence'], 'spearman'):.2f}. "
        "Arousal and dominance are practically one dimension here: vocal activation (energetic / forceful vs flat). "
        "MSP-Podcast listeners' own arousal and dominance ratings are strongly correlated, and the model inherits that. "
        "So the measure can say 'flatter / less forceful than your usual', not 'less confident' as something separate "
        "from energy, and `tension_rise` cannot tell tension from enthusiasm.\n")
    add("\n## 3. What else does it track? (train, Spearman with answer-level descriptors)\n")
    add("A dominance score that mainly follows loudness / microphone, length or speech rate would be a confound, not "
        "a new measurement.\n")
    others = [c for c in ["intensity_mean_db", "intensity_std_db", "f0_median_st", "pitch_std_st", "speech_rate_wpm",
                          "artic_rate_sps", "mid_pause_rate_per_min", "audio_duration_s", "jitter_local", "shimmer_local",
                          "asr_prob_mean"] if c in tr]
    add("| | " + " | ".join(f"`{c}`" for c in others) + " |")
    add("|---|" + "---|" * len(others))
    for a in ATTRS:
        add(f"| `ser_{a}_mean` | " + " | ".join(f"{spearmanr(tr[f'ser_{a}_mean'], tr[c], nan_policy='omit')[0]:+.2f}" for c in others) + " |")
    vq = tr.groupby("video_quality")["ser_dominance_mean"].median()
    add("\nMedian dominance by video quality (proxy for recording setup): " + ", ".join(f"{k}: {v:.3f}" for k, v in vq.items()) + ".\n")
    add("## 4. Moments within answers (for flags)\n")
    for c, label in (("ser_dominance_drop_sd", "largest dominance drop (speaker SD)"), ("ser_tension_run_s", "longest tension run (s)")):
        v = u[c].dropna()
        add(f"- {label}: median {v.median():.2f}, p90 {v.quantile(.9):.2f}, share > 0 / > 1: "
            f"{(v > 0).mean():.0%} / {(v > 1).mean():.0%}.")
    add("\nFlags, percentiles and the composure class come from Stage 2 (`stage2_reference/report.md`); validation against "
        "the rater scores from Stage 3 (`stage3_recruitview_validation/decisions.csv`, rows `ser_*`, `composure_index`, "
        "`flag_low_dominance`, `flag_tension_rise`, `flag_composure_drop`).\n")
    add("## 5. Not done\n")
    add("- **Second SER model (agreement check).** `3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes` loads only with "
        "`trust_remote_code=True` (runs Python from a third-party repository); not run without explicit approval. "
        "Until then there is no model-agreement check, which is one of the stage's promotion criteria.")
    add("- **MIT validation** (`Calm`, `NotStressed`, `Overall`): Stage 7. `run_ser_windows.py` works on any clip list.\n")
    P.S5_REPORT.write_text("\n".join(L), encoding="utf-8")


# ============================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="audeering")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    seg = pd.read_csv(P.SEGMENTS, dtype={"id": str, "participant": str, "duplicate_of": str})
    keys = seg[["id", "participant", "duplicate_of"]].assign(group=lambda d: d["duplicate_of"].fillna(d["id"]))
    windows = pd.read_csv(P.s5_windows(args.model), dtype={"id": str})
    windows = windows[[c for c in windows.columns if not c.endswith("_z") and c != "f0_st"]]
    clips = X.load_clips(None).merge(keys[["id"]], on="id")
    settings = pd.read_csv(P.PITCH_SETTINGS, dtype={"participant": str}).set_index("participant")
    windows["f0_st"] = window_pitch(windows, clips, settings, args.workers)
    for col in ("arousal", "dominance", "valence", "f0_st"):
        windows[f"{'f0' if col == 'f0_st' else col}_z"] = speaker_z(windows.reset_index(drop=True), keys, col)
    windows.to_csv(P.s5_windows(args.model), index=False)

    rows = [{"id": cid, **answer_measures(g)} for cid, g in windows.groupby("id")]
    ser = keys[["id"]].merge(pd.DataFrame(rows), on="id", how="left")
    ser.to_csv(P.S5_SEGMENTS, index=False)
    P.S5_REGISTRY.write_text(json.dumps({"_source": f"Stage 5 SER ({args.model})", "descriptors": registry(args.model)},
                                        indent=2), encoding="utf-8")
    write_report(args.model, windows, ser, seg)
    print(f"Done: {len(ser)} answers. Report: {P.S5_REPORT}")


if __name__ == "__main__":
    main()
