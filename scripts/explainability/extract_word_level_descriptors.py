"""Stage 4, step 2: re-measure every delivery descriptor on REAL sentences from the verbatim ASR.

What changes against Stage 1 (and why)
  sentences      Stage 1 used the RecruitView transcript lines. Those are subtitle lines, not sentences: they
                 break mid-sentence ("I try not to let | the stress get to me"), so trailing off and terminal
                 rises were partly measured at points that are not sentence ends. Here a sentence ends at
                 Whisper's sentence punctuation (. ? !).
  word counts    from the verbatim ASR, not the cleaned transcript (which drops words in slow speech and
                 sometimes repeats lines). Filled pauses (um, uh, ...) are NOT counted as words.
  pause classes  a VAD pause is mid-sentence when the words on both sides of it belong to the same sentence
                 (word timestamps), instead of the +-0.5 s zones around 1-s transcript stamps.
  new columns    filled pauses (diagnostic: counting and feedback on fillers belong to the text branch),
                 ASR word confidence (experimental intelligibility proxy), ASR quality checks.
Everything else (VAD speech, Praat pitch / intensity, every formula) is reused unchanged from Stage 1
(`extract_delivery_descriptors.py`, cached Praat frames), so differences come only from words and sentences.

Inputs   results/explainability/stage4_asr/words.csv (run_verbatim_asr.py), Stage 1 tables (keys, duplicates),
         speaker pitch settings, cached Praat frames, VAD regions
Outputs  results/explainability/stage4_asr/
           tables/{sentence_descriptors, segment_descriptors, pause_events}.csv   same schema as Stage 1 + new columns
           descriptor_registry.json, report.md, listening_check/, qc_plots/
Stages 2-3 read these tables automatically (paths.DESCRIPTOR_SOURCE == "asr").

Run:  python scripts/explainability/extract_word_level_descriptors.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import extract_delivery_descriptors as X  # noqa: E402  (Stage 1 measurement code, reused unchanged)

FILLED = {"um", "umm", "ummm", "uh", "uhh", "uhhh", "uhm", "erm", "er", "err", "ah", "ahh", "hmm", "hm", "mm", "mmm", "eh"}
END_RE = re.compile(r"[.?!][\"')\]]*$")
ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "e.g.", "i.e.", "etc.", "vs.", "st.", "u.s.", "no."}
REPEAT_NGRAM = 4
REPEAT_MIN = 4               # the same 4-gram >= 4 times in one answer = likely Whisper repetition loop
TOKEN_RUN_MAX = 5            # ... or the same token >= 5 times in a row
WINDOW_PAD_START_S = 0.2
WINDOW_PAD_END_S = 0.5
RATIO_LOW, RATIO_HIGH = 0.7, 1.4  # ASR / transcript word ratio outside this = the two disagree, listen
KEY_COLUMNS = ["id", "split", "participant", "question_id", "question_type", "duration_bucket", "video_quality",
               "is_duplicate", "duplicate_of", "duplicate_group_size"]


# ============================================================================================
# Sentences from words
# ============================================================================================
def build_sentences(words: pd.DataFrame, audio_duration: float) -> list[dict]:
    """Group a clip's ASR words (sorted, hallucination-filtered) into sentences at . ? ! punctuation."""
    groups, cur = [], []
    for w in words.itertuples(index=False):
        cur.append(w)
        tok = str(w.word).strip().lower()
        if END_RE.search(tok) and tok not in ABBREVIATIONS:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    sentences = []
    for i, g in enumerate(groups):
        content = [w for w in g if not w.is_filled_pause]
        tokens = [t for w in content for t in X.WORD_RE.findall(str(w.word))]
        # Whisper word edges are only ~0.1-0.3 s accurate and often cut the last syllable short. Pad the window
        # (never into the neighbouring sentence's words) and let the VAD trimming find the real speech edges,
        # as Stage 1 does with the 1-s stamps.
        prev_end = float(groups[i - 1][-1].end) if i > 0 else 0.0
        next_start = float(groups[i + 1][0].start) if i + 1 < len(groups) else audio_duration
        start = max(prev_end, float(g[0].start) - WINDOW_PAD_START_S, 0.0)
        end = min(max(next_start, float(g[-1].end)), float(g[-1].end) + WINDOW_PAD_END_S, audio_duration)
        probs = [w.prob for w in content]
        sentences.append({
            "sent_idx": i, "ts_start": start, "ts_end": end, "win_start": start, "win_end": max(start, end),
            "text": " ".join(str(w.word).strip() for w in g),
            "n_words": sum(word_units(str(w.word)) for w in content),
            "n_syllables": sum(X.count_syllables(t) for t in tokens),
            "word_mids": [(w.start + w.end) / 2 for w in g],
            "n_filled_pauses": len(g) - len(content),
            "asr_prob_mean": float(np.mean(probs)) if probs else np.nan,
        })
    return sentences


def repetition_suspect(words: list[str]) -> bool:
    """Whisper repetition loop: the same 4-gram >= 4 times, or the same token >= 5 times in a row (e.g. 30 x "uh"
    written instead of the real words). All tokens, fillers included: filler loops are the common case."""
    toks = [t for t in (re.sub(r"[^a-z0-9']", "", w.lower()) for w in words) if t]
    grams = [" ".join(toks[i:i + REPEAT_NGRAM]) for i in range(len(toks) - REPEAT_NGRAM + 1)]
    if grams and pd.Series(grams).value_counts().iloc[0] >= REPEAT_MIN:
        return True
    run = best = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best >= TOKEN_RUN_MAX


def word_units(word: str) -> int:
    """Spoken words in an ASR token. Whisper writes numbers as digits: a long digit string (phone number, ID) is read
    digit by digit; a short number ("2022", "500") is ~2 spoken words."""
    tokens = X.WORD_RE.findall(word)
    n = 0
    for t in tokens:
        if t.isdigit():
            n += len(t) if len(t) > 4 else 2 if len(t) >= 3 else 1
        else:
            n += 1
    return n


RATE_COLUMNS_SENT = ["speech_rate_wpm", "artic_rate_sps", "artic_rate_wpm"]
RATE_COLUMNS_SEG = ["speech_rate_wpm", "artic_rate_sps", "artic_rate_wpm", "sent_rate_mean_wpm", "sent_rate_max_wpm",
                    "sent_rate_std_wpm", "sent_rate_cv", "rushed_opening_ratio", "rate_drift_ratio",
                    "filled_pause_count", "filled_pause_rate_per_min"]


# ============================================================================================
# Registry (Stage 1 definitions + what changed)
# ============================================================================================
def stage4_registry() -> dict:
    reg = json.loads(json.dumps(X.REGISTRY))
    src = " [Stage 4: words from verbatim ASR (faster-whisper medium.en), filled pauses excluded; sentences end at ASR punctuation.]"
    for key in ("sentence.speech_rate_wpm", "sentence.artic_rate_sps", "sentence.artic_rate_wpm", "sentence.n_words",
                "segment.speech_rate_wpm", "segment.artic_rate_sps", "segment.artic_rate_wpm", "segment.word_count",
                "segment.syllable_count", "segment.n_sentences", "segment.sent_rate_mean_wpm", "segment.sent_rate_max_wpm",
                "segment.sent_rate_std_wpm", "segment.sent_rate_cv", "segment.rushed_opening_ratio", "segment.rate_drift_ratio"):
        reg[key]["definition"] = reg[key]["definition"].replace("cleaned transcript", "verbatim ASR").replace(
            "Timestamped transcript sentences", "ASR sentences") + src
    for key in ("sentence.n_mid_pauses", "segment.mid_pause_rate_per_min", "segment.long_mid_pause_rate_per_min",
                "segment.mid_pause_max_s", "segment.mid_pause_time_ratio", "segment.between_pause_time_ratio"):
        reg[key]["definition"] += " [Stage 4: mid-sentence = the ASR words on both sides of the pause belong to the same sentence.]"
    for key in ("sentence.end_drop_db", "sentence.terminal_slope_st_s"):
        reg[key]["definition"] += " [Stage 4: measured at real sentence ends (ASR punctuation), not transcript line breaks.]"

    def add(level, name, category, status, unit, definition):
        reg[f"{level}.{name}"] = {"level": level, "name": name, "category": category, "status": status,
                                  "unit": unit, "definition": definition}
    fp_note = ("Filler counting and filler feedback belong to the TEXT branch (it receives words.csv); kept here only to "
               "check the ASR and to exclude fillers from speech rate.")
    add("sentence", "n_filled_pauses", "pausing", X.D, "count", "um / uh / erm / hmm tokens in the sentence. " + fp_note)
    add("sentence", "asr_prob_mean", "projection", X.E, "probability",
        "Mean Whisper word probability (filled pauses excluded): intelligibility proxy. Also drops with noise and "
        "accent, so it waits for the Stage 6 quality gate and Stage 7-8 validation.")
    add("segment", "filled_pause_count", "pausing", X.D, "count", "um / uh / erm / hmm tokens in the answer. " + fp_note)
    add("segment", "filled_pause_rate_per_min", "pausing", X.D, "per min", "Filled pauses per minute of speech span. " + fp_note)
    add("segment", "asr_prob_mean", "projection", X.E, "probability", "Mean Whisper word probability over the answer.")
    add("segment", "asr_words_outside_vad", "diagnostic", X.D, "count",
        "ASR words with no VAD speech within 0.1 s (likely hallucinations); removed before measuring.")
    add("segment", "transcript_word_count", "diagnostic", X.D, "count", "Words in the cleaned RecruitView transcript (Stage 1).")
    add("segment", "asr_transcript_word_ratio", "diagnostic", X.D, "ratio",
        f"ASR words / transcript words. Outside [{RATIO_LOW}, {RATIO_HIGH}] = the two disagree; listen before trusting either.")
    add("segment", "asr_repetition_suspect", "diagnostic", X.D, "bool",
        f"The same {REPEAT_NGRAM}-word sequence occurs >= {REPEAT_MIN} times, or one token >= {TOKEN_RUN_MAX} times in a "
        "row (Whisper repetition loop). All word-based rates of the answer are then withheld.")
    return reg


# ============================================================================================
# Report
# ============================================================================================
def write_report(seg: pd.DataFrame, sent: pd.DataFrame, pauses: pd.DataFrame, s1_seg: pd.DataFrame,
                 s1_sent: pd.DataFrame, s1_pauses: pd.DataFrame, words: pd.DataFrame, n_hallucinated: int,
                 line_ends: dict, slow_check: dict, term: dict) -> Path:
    L = []
    add = L.append
    u = seg[~seg["is_duplicate"]]
    add("# Stage 4: verbatim ASR and word-level descriptors\n")
    add("Generated by `scripts/explainability/extract_word_level_descriptors.py` (ASR: `run_verbatim_asr.py`, "
        "faster-whisper medium.en, word timestamps, filler-preserving prompt). Same measurement code as Stage 1; only the "
        "words and the sentence boundaries differ. Statistics on unique recordings unless stated.\n")

    add("## 1. ASR coverage and quality\n")
    ratio = u["asr_transcript_word_ratio"]
    add(f"- Clips with ASR words: {(seg['word_count'] > 0).sum()} / {len(seg)}; words: {len(words):,}; "
        f"filled pauses: {int(words['is_filled_pause'].sum()):,} ({words['is_filled_pause'].mean():.1%} of tokens).")
    add(f"- Words removed as possible hallucinations (no VAD speech within 0.1 s): {n_hallucinated} "
        f"in {(seg['asr_words_outside_vad'] > 0).sum()} clips.")
    add(f"- Possible repetition loops: {int(seg['asr_repetition_suspect'].sum())} clips (listed in `tables/asr_checks.csv`).")
    add(f"- ASR / transcript word ratio: median {ratio.median():.2f}, p5 {ratio.quantile(.05):.2f}, p95 {ratio.quantile(.95):.2f}; "
        f"{((ratio < RATIO_LOW) | (ratio > RATIO_HIGH)).sum()} unique clips outside [{RATIO_LOW}, {RATIO_HIGH}] (`tables/asr_checks.csv`).")
    add(f"- Median word probability: {u['asr_prob_mean'].median():.3f}.\n")

    add("## 2. Sentences: transcript lines vs real sentences\n")
    add(f"- Transcript lines (Stage 1): {len(s1_sent):,}; ASR sentences: {len(sent):,}.")
    add(f"- Median duration: transcript line {s1_sent['duration_s'].median():.1f} s, ASR sentence {sent['duration_s'].median():.1f} s "
        f"(p95 {sent['duration_s'].quantile(.95):.1f} s).")
    add(f"- Only **{line_ends['share']:.0%}** of transcript line ends fall within 0.5 s of a real sentence end "
        f"({line_ends['n']:,} line ends checked). The rest are subtitle breaks inside a sentence, where Stage 1 measured "
        "trailing off and terminal rises anyway.\n")

    add("## 3. What changed in the measurements (answer level, unique recordings)\n")
    m = u.merge(s1_seg[["id"] + COMPARE], on="id", suffixes=("", "_s1"))
    add("| Descriptor | Spearman (transcript vs ASR) | Median transcript | Median ASR | Missing transcript | Missing ASR |")
    add("|---|---|---|---|---|---|")
    for c in COMPARE:
        ok = m[[c, f"{c}_s1"]].dropna()
        rho = spearmanr(ok[c], ok[f"{c}_s1"])[0] if len(ok) > 10 else np.nan
        add(f"| `{c}` | {rho:.2f} | {m[f'{c}_s1'].median():.3g} | {m[c].median():.3g} | "
            f"{m[f'{c}_s1'].isna().mean():.1%} | {m[c].isna().mean():.1%} |")
    slow = m[m["speech_rate_wpm_s1"] <= m["speech_rate_wpm_s1"].quantile(.05)]
    fast = m[m["speech_rate_wpm_s1"] >= m["speech_rate_wpm_s1"].quantile(.95)]
    add(f"\nThe slowest 5% of answers by transcript rate have a median ASR / transcript word ratio of "
        f"**{slow['asr_transcript_word_ratio'].median():.2f}** (fastest 5%: {fast['asr_transcript_word_ratio'].median():.2f}); "
        f"their median rate moves from {slow['speech_rate_wpm_s1'].median():.0f} to {slow['speech_rate_wpm'].median():.0f} WPM. "
        + ("The cleaned transcript dropped words where it made answers look slow.\n"
           if slow["asr_transcript_word_ratio"].median() > 1.1 else
           "Over whole answers the transcript and the ASR count about the same words: the omissions found in Stage 1 "
           "are local (single lines), so they matter for sentence rates and the slow-sentence flag, little for answer rates.\n"))

    add("### Slow side (decides whether `too_slow` can return)\n")
    add("For the slowest / fastest 25 rate-valid ASR sentences (unique recordings): transcript words falling in the same time "
        "span (pro-rated by overlap with the transcript lines) and words from a second ASR model (faster-whisper small.en, "
        "run on these clips). If neither finds clearly more words than medium.en, the slow values are not caused by "
        "missing words.\n")
    add("| Group | Median WPM | Median artic. (syll/s) | ASR / transcript words | medium / small.en words (n) |")
    add("|---|---|---|---|---|")
    for name, row in slow_check.items():
        add(f"| {name} | {row['wpm']:.0f} | {row['artic']:.2f} | {row['ratio_transcript']:.2f} | "
            f"{row['ratio_small']:.2f} ({row['n_small']}) |")
    add("\nThe sentences themselves are in `listening_check/listening_check.csv` (fastest / slowest sentence).\n")

    add("### Terminal pitch at real sentence ends\n")
    add(f"- Last sentence of each answer (a real end in both stages): median terminal slope Stage 1 "
        f"{term['final_s1']:.2f} st/s, Stage 4 {term['final_s4']:.2f} st/s, Spearman {term['final_rho']:.2f}: consistent.")
    add(f"- Sentence ends **inside** an answer: median {term['inner_s4']:+.2f} st/s (rising). In spontaneous speech a "
        "sentence that is not the end of the turn usually ends with a continuation rise (\"...and then\"). A terminal rise "
        "therefore does not mean uptalk; `uptalk` stays experimental and Stage 8 should label rising *statements* "
        "separately from continuation rises.")
    add(f"- End-of-sentence energy drop is larger at real ends (median {term['drop_s4']:.1f} dB vs {term['drop_s1']:.1f} dB at "
        "transcript line breaks): the voice normally falls at a sentence end, so the `trails_off` reference must be (and is) "
        "refitted on Stage 4 sentences.\n")

    add("## 4. Pause classes\n")
    add("| Class | Stage 1 (transcript stamps) | Stage 4 (word boundaries) |")
    add("|---|---|---|")
    c1, c4 = s1_pauses[s1_pauses["counted"]]["pause_class"].value_counts(), pauses[pauses["counted"]]["pause_class"].value_counts()
    for cls in sorted(set(c1.index) | set(c4.index)):
        add(f"| {cls} | {c1.get(cls, 0):,} | {c4.get(cls, 0):,} |")
    add("\nCounted pauses (>= 0.25 s) only. The VAD pauses are identical; only their class changes.\n")

    add("## 5. Filled pauses (for the text branch)\n")
    add(f"- Answers with at least one filled pause: {(u['filled_pause_count'] > 0).mean():.0%}; median rate "
        f"{u['filled_pause_rate_per_min'].median():.2f}/min, p90 {u['filled_pause_rate_per_min'].quantile(.9):.2f}/min.")
    add("- Whisper under-transcribes fillers even with the prompt; recall has not been hand-checked. Filler counting, "
        "the filler hand-check (precision/recall on 30 clips) and filler feedback are the text branch's job; this branch "
        "uses filled pauses only to keep them out of the word counts.\n")

    add("## 6. Missing values\n")
    nan4 = X.nan_table(sent, seg).set_index("descriptor")["nan_pct"]
    nan1 = X.nan_table(s1_sent, s1_seg).set_index("descriptor")["nan_pct"]
    add("Sentence-level values over rate-valid sentences, answer-level over all answers. Non-diagnostic descriptors only.\n")
    add("| Descriptor | Status | Missing, Stage 1 | Missing, Stage 4 |")
    add("|---|---|---|---|")
    for key, meta in X.REGISTRY.items():
        if meta["status"] != X.D and key in nan4.index:
            add(f"| `{key}` | {meta['status']} | {nan1.get(key, np.nan):.1f}% | {nan4[key]:.1f}% |")
    add("\n## 7. Listening check\n")
    add("`listening_check/listening_check.csv` lists the five most extreme sentences/answers per check, now on real "
        "sentences. QC plots for the most extreme clip per check: `qc_plots/`.\n")
    P.S4_REPORT.write_text("\n".join(L), encoding="utf-8")
    return P.S4_REPORT


COMPARE = ["word_count", "n_sentences", "speech_rate_wpm", "artic_rate_sps", "sent_rate_mean_wpm", "sent_rate_cv",
           "mid_pause_rate_per_min", "mid_pause_max_s", "end_drop_median_db", "terminal_slope_median_st_s"]


def slow_side_check(sent: pd.DataFrame, s1_sent: pd.DataFrame, clips: pd.DataFrame, n: int = 25) -> dict:
    valid = sent[sent["rate_valid"] & ~sent["is_duplicate"]].dropna(subset=["speech_rate_wpm"])
    lines = {k: g for k, g in s1_sent.groupby("id")}
    extreme_ids = sorted(set(valid.nsmallest(n, "speech_rate_wpm")["id"]) | set(valid.nlargest(n, "speech_rate_wpm")["id"]))
    import verify_stage1 as V  # second, independent ASR model (small.en; cached)
    V.whisper_words(extreme_ids, clips.set_index("id", drop=False), "small.en")
    small = {}
    out = {}
    for name, pick in (("slowest 25", valid.nsmallest(n, "speech_rate_wpm")), ("fastest 25", valid.nlargest(n, "speech_rate_wpm"))):
        r_t, r_s = [], []
        for r in pick.itertuples():
            g = lines.get(r.id)
            if g is not None:
                ov = np.clip(np.minimum(r.speech_end, g["speech_end"]) - np.maximum(r.speech_start, g["speech_start"]), 0, None)
                tw = float(np.sum(g["n_words"] * ov / g["duration_s"].where(g["duration_s"] > 0, np.inf)))
                if tw > 0:
                    r_t.append(r.n_words / tw)
            cache = P.CACHE_WHISPER / f"{r.id}_small.en.json"
            if cache.exists():
                small.setdefault(r.id, json.loads(cache.read_text(encoding="utf-8")))
                ws = [w for w in small[r.id] if r.speech_start <= (w["start"] + w["end"]) / 2 <= r.speech_end
                      and re.sub(r"[^a-z]", "", w["w"].lower()) not in FILLED]
                if ws:
                    r_s.append(r.n_words / len(ws))
        out[name] = {"wpm": pick["speech_rate_wpm"].median(), "artic": pick["artic_rate_sps"].median(),
                     "ratio_transcript": np.median(r_t) if r_t else np.nan,
                     "ratio_small": np.median(r_s) if r_s else np.nan, "n_small": len(r_s)}
    return out


def terminal_check(sent: pd.DataFrame, s1_sent: pd.DataFrame) -> dict:
    s1 = s1_sent[s1_sent["id"].isin(set(sent["id"]))]
    l4 = sent.sort_values("sent_idx").groupby("id").tail(1)
    l1 = s1.sort_values("sent_idx").groupby("id").tail(1)
    m = l4[["id", "terminal_slope_st_s"]].merge(l1[["id", "terminal_slope_st_s"]], on="id", suffixes=("_4", "_1")).dropna()
    return {"final_s1": m["terminal_slope_st_s_1"].median(), "final_s4": m["terminal_slope_st_s_4"].median(),
            "final_rho": spearmanr(m["terminal_slope_st_s_1"], m["terminal_slope_st_s_4"])[0],
            "inner_s4": sent.drop(l4.index)["terminal_slope_st_s"].median(),
            "drop_s4": sent["end_drop_db"].median(), "drop_s1": s1["end_drop_db"].median()}


def line_end_agreement(s1_sent: pd.DataFrame, sent: pd.DataFrame, tol: float = 0.5) -> dict:
    """Share of transcript line ends (excluding each clip's last line) within tol of an ASR sentence end."""
    ends = sent.groupby("id")["speech_end"].apply(np.asarray).to_dict()
    hits = n = 0
    for cid, g in s1_sent.groupby("id"):
        e = ends.get(cid)
        if e is None or not len(e):
            continue
        for v in g.sort_values("sent_idx")["speech_end"].iloc[:-1]:
            n += 1
            hits += bool(e is not None and len(e) and np.min(np.abs(e - v)) <= tol)
    return {"share": hits / max(n, 1), "n": n}


# ============================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for d in (P.S4_TABLES, P.S4_LISTENING, P.S4_QC_PLOTS):
        d.mkdir(parents=True, exist_ok=True)

    dt = {"id": str, "participant": str, "duplicate_of": str}
    s1_seg = pd.read_csv(P.S1_SEGMENTS, dtype=dt)
    s1_sent = pd.read_csv(P.S1_SENTENCES, dtype=dt)
    s1_pauses = pd.read_csv(P.S1_PAUSES, dtype=dt)
    # keep_default_na=False: spoken words such as "null" / "NA" / "none" must stay words, not NaN
    words = pd.read_csv(P.WORDS, dtype={"id": str, "word": str}, keep_default_na=False).sort_values(["id", "start", "word_idx"])
    clips = X.load_clips(None)
    clips = clips[clips["id"].isin(s1_seg["id"])].reset_index(drop=True)
    vad, blips = X.load_vad(set(clips["id"]))
    settings = pd.read_csv(P.PITCH_SETTINGS, dtype={"participant": str}).set_index("participant")

    print("Loading cached Praat frames...")
    jobs = [(r.id, r.audio_path, settings.loc[r.participant, "pitch_floor_hz"], settings.loc[r.participant, "pitch_ceiling_hz"])
            for r in clips.itertuples()]
    frames = X.run_parallel(X._pass2, jobs, args.workers, "pass2 (cached)")

    print("Computing descriptors on ASR sentences...")
    by_clip = {k: g for k, g in words.groupby("id")}
    tw = s1_seg.set_index("id")["word_count"]
    sent_rows, seg_rows, pause_rows, n_halluc = [], [], [], 0
    for clip in clips.itertuples(index=False):
        clip = pd.Series(clip._asdict())
        cid = clip["id"]
        w = by_clip.get(cid, words.iloc[:0])
        kept = w[w["in_vad_speech"]]
        n_out = int((~w["in_vad_speech"]).sum())
        n_halluc += n_out
        sentences = build_sentences(kept, clip["audio_duration_s"])
        s, a, p = X.describe_clip(clip, vad[cid], frames[cid], settings.loc[clip["participant"], "reference_f0_hz"],
                                  settings.loc[clip["participant"]], blips[cid], sentences=sentences)
        for row, sd in zip(s, sentences):
            row["n_filled_pauses"] = sd["n_filled_pauses"]
            row["asr_prob_mean"] = sd["asr_prob_mean"]
        content = kept[~kept["is_filled_pause"]]
        n_fp = int(kept["is_filled_pause"].sum())
        span = a["speech_span_s"]
        a.update({
            "filled_pause_count": n_fp,
            "filled_pause_rate_per_min": 60 * n_fp / span if np.isfinite(span) and span > 0 else np.nan,
            "asr_prob_mean": float(content["prob"].mean()) if len(content) else np.nan,
            "asr_words_outside_vad": n_out,
            "transcript_word_count": int(tw.get(cid, 0)),
            "asr_transcript_word_ratio": a["word_count"] / tw[cid] if tw.get(cid, 0) > 0 else np.nan,
            "asr_repetition_suspect": repetition_suspect(kept["word"].astype(str).tolist()),
        })
        if a["asr_repetition_suspect"]:
            # A loop replaces real words, so word-based rates (and filler counts) are not measurements of the
            # speaker: withhold them, as Stage 1 does for transcript lines that do not match their timestamps.
            for row in s:
                row.update({c: np.nan for c in RATE_COLUMNS_SENT})
                row["rate_valid"], row["timing_suspect"] = False, True
            a.update({c: np.nan for c in RATE_COLUMNS_SEG})
            a["transcript_timing_suspect"] = True
        sent_rows += s
        seg_rows.append(a)
        pause_rows += p

    keys = s1_seg[KEY_COLUMNS]
    sent = keys.merge(pd.DataFrame(sent_rows), on="id")
    seg = keys.merge(pd.DataFrame(seg_rows), on="id")
    pauses = keys[["id", "split", "participant"]].merge(pd.DataFrame(pause_rows), on="id")
    sent.to_csv(P.S4_SENTENCES, index=False)
    seg.to_csv(P.S4_SEGMENTS, index=False)
    pauses.to_csv(P.S4_PAUSES, index=False)
    checks = seg[(seg["asr_repetition_suspect"]) | (seg["asr_transcript_word_ratio"] < RATIO_LOW)
                 | (seg["asr_transcript_word_ratio"] > RATIO_HIGH) | (seg["asr_words_outside_vad"] > 0)]
    checks[["id", "split", "is_duplicate", "audio_duration_s", "word_count", "transcript_word_count",
            "asr_transcript_word_ratio", "asr_words_outside_vad", "asr_repetition_suspect", "asr_prob_mean"]] \
        .to_csv(P.S4_TABLES / "asr_checks.csv", index=False)
    s1_registry = json.loads(P.S1_REGISTRY.read_text(encoding="utf-8"))
    s1_registry["descriptors"] = stage4_registry()
    s1_registry["_source"] = "Stage 4: verbatim ASR words and sentences (extract_word_level_descriptors.py)"
    P.S4_REGISTRY.write_text(json.dumps(s1_registry, indent=2), encoding="utf-8")

    for old in P.S4_QC_PLOTS.glob("*.png"):
        old.unlink()
    sheet = X.listening_sheet(clips, sent, seg, pauses)
    sheet.to_csv(P.S4_LISTENING / "listening_check.csv", index=False)
    by_id = clips.set_index("id")
    for check in sheet["check"].unique():
        cid = sheet[sheet["check"] == check].iloc[0]["id"]
        X.qc_plot(by_id.loc[cid], vad[cid], frames[cid], settings.loc[by_id.loc[cid, "participant"], "reference_f0_hz"],
                  sent[sent.id == cid], pauses[pauses.id == cid],
                  P.S4_QC_PLOTS / f"{check.replace(' ', '_')}_{cid}.png", f"{check}: clip {cid} (ASR sentences)")

    report = write_report(seg, sent, pauses, s1_seg, s1_sent, s1_pauses, words, n_halluc,
                          line_end_agreement(s1_sent, sent), slow_side_check(sent, s1_sent, clips),terminal_check(sent, s1_sent))
    print(f"Done. {len(sent)} sentences, {len(seg)} answers. Report: {report}")


if __name__ == "__main__":
    main()
