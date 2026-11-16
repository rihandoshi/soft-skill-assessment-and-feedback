"""Phase 6: hand the audio branch's per-answer outputs to the fusion model (late fusion). CPU only, about 5 min.

One row per answer for every RecruitView answer (2,011) and every MIT answer (674), keyed by the dataset answer id
and the shared person-level split (Datasets/Split Dataset, `user_no`):

  audio score predictions   pred_<target> for the 12 RecruitView targets. WavLM-base-plus layer 7 (mean + std pooled)
                            + Ridge, the best audio score model in the repeated CV (results/audio_score_models).
                            Train answers: OUT-OF-FOLD (5-fold GroupKFold over people), so a fusion model trained on
                            train never sees an audio prediction made by a model that saw that answer's labels.
                            Val / test / MIT: one model fitted on all train people. Fitted on unique recordings
                            (duplicate copies averaged), alpha chosen by grouped CV. Same recipe as
                            run_confidence_detector.py, extended to every target.
  feedback dimensions       hesitation / expressiveness index (0-100) and level, withheld flags, the four hesitation
                            components and activation (fit_answer_levels.py). No labels are used for these, so they
                            are not out-of-fold and need not be
  context                   speaking time, recording quality, speaker pitch (internal: never shown to a user)

Also: the WavLM layer-7 embedding per answer (for embedding-level fusion), the evidence JSON (Stage 9), and SCHEMA.md.

Output  results/explainability/stage13_fusion_export/{audio_answers.csv, audio_embeddings_wavlm_l7.npz,
        evidence.jsonl, score_model_metrics.csv, SCHEMA.md}
Run:    python scripts/explainability/fit_answer_levels.py && python scripts/explainability/build_answer_evidence.py
        python scripts/explainability/export_for_fusion.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths as P  # noqa: E402
import run_confidence_detector as RC  # noqa: E402  (embeddings, Ridge recipe)

TARGETS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism", "overall_personality",
           "interview_score", "answer_score", "speaking_skills", "confidence_score", "facial_expression",
           "overall_performance"]
LEVEL_COLS = ["hesitation_index", "hesitation_level", "hesitation_withheld", "hesitation_reasons",
              "expressiveness_index", "expressiveness_level", "expressiveness_withheld", "expressiveness_reasons"]
COMPONENT_COLS = ["p_interjection", "p_repetition_block", "sf_p_disfluent", "long_mid_pause_rate_per_min",
                  "activation", "activation_adj", "pitch_std_st"]
CONTEXT_COLS = ["speech_span_s", "acoustic_ok", "speaker_f0_hz"]


def split_table() -> pd.DataFrame:
    return pd.concat([pd.read_csv(RC.SPLIT_DIR / f"{s}.csv", dtype={"id": str},
                                  usecols=["id", "question_id", "user_no", *TARGETS]).assign(split=s)
                      for s in ("train", "val", "test")], ignore_index=True)


def score_predictions(rv: pd.DataFrame, X_rv: np.ndarray, X_mit: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """pred_<target> for RecruitView (train out-of-fold) and MIT, and the metrics per target."""
    train = rv[rv["split"] == "train"]
    uniq = train[~train["is_duplicate"]]
    Xu, groups = X_rv[uniq.index.to_numpy()], uniq["participant"].to_numpy()
    is_train = (rv["split"] == "train").to_numpy()
    pred_rv, pred_mit, metrics = pd.DataFrame(index=rv.index), {}, []
    for t in TARGETS:
        y = uniq["duplicate_of"].map(train.groupby("duplicate_of")[t].mean()).to_numpy(float)
        cv = {a: spearmanr(RC.grouped_oof(Xu, y, groups, a), y)[0] for a in RC.ALPHAS}
        alpha = max(cv, key=cv.get)
        oof = pd.Series(RC.grouped_oof(Xu, y, groups, alpha), index=uniq["duplicate_of"].to_numpy())
        final = RC.model(alpha).fit(Xu, y)
        p = final.predict(X_rv)
        p[is_train] = rv.loc[is_train, "duplicate_of"].map(oof).to_numpy(float)
        pred_rv[f"pred_{t}"] = p
        pred_mit[f"pred_{t}"] = final.predict(X_mit)
        row = {"target": t, "alpha": alpha, "train_oof_spearman": cv[alpha]}
        for s in ("val", "test"):
            m = (rv["split"] == s).to_numpy()
            row[f"{s}_spearman"] = spearmanr(p[m], rv.loc[m, t])[0]
        metrics.append(row)
        print(f"  {t:20s} alpha {alpha:>8g}  train OOF {row['train_oof_spearman']:.3f}  val {row['val_spearman']:.3f}"
              f"  test {row['test_spearman']:.3f}", flush=True)
    return pred_rv, pd.DataFrame(pred_mit), pd.DataFrame(metrics)


def main() -> None:
    out = P.S13
    out.mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(P.SEGMENTS, dtype={"id": str, "duplicate_of": str},
                      usecols=["id", "participant", "is_duplicate", "duplicate_of"])
    rv = split_table().merge(seg, on="id", how="left")
    assert rv["participant"].notna().all(), "RecruitView answers without audio outputs"
    rv["is_duplicate"] = rv["is_duplicate"].astype(bool)
    print("audio score models (WavLM L7 + Ridge):", flush=True)
    X_rv = RC.rv_embeddings(rv)
    mit, X_mit = RC.mit_embeddings()
    pred_rv, pred_mit, metrics = score_predictions(rv, X_rv, X_mit)

    lv = pd.read_csv(P.ANSWER_LEVELS, dtype={"id": str}, low_memory=False)
    quality = pd.concat([pd.read_csv(P.S6_TABLES / "quality_gate_per_clip.csv", dtype={"id": str}),
                         pd.read_csv(P.S7_TABLES / "mit_segments.csv", dtype={"id": str})], ignore_index=True)
    quality = quality.drop_duplicates("id").set_index("id")["quality_composite"] \
        if "quality_composite" in quality else pd.Series(dtype=float)
    keep = ["id", "setting", *LEVEL_COLS, *COMPONENT_COLS, *CONTEXT_COLS]

    a = pd.concat([rv[["id"]].join(pred_rv), mit[["id"]].join(pred_mit)], ignore_index=True)
    keys = pd.concat([
        pd.DataFrame({"answer_id": rv["id"], "dataset": "recruitview", "split": rv["split"],
                      "person_id": rv["user_no"].astype(str), "question_id": rv["question_id"].astype(str),
                      "is_duplicate": rv["is_duplicate"], "duplicate_of": rv["duplicate_of"]}),
        pd.DataFrame({"answer_id": mit["id"], "dataset": "mit", "split": "mit",
                      "person_id": mit["participant"].astype(str), "question_id": mit["question"],
                      "is_duplicate": False, "duplicate_of": mit["id"],
                      "interview_id": mit["interview"], "session": mit["session"]})], ignore_index=True)
    table = keys.merge(a.rename(columns={"id": "answer_id"}), on="answer_id", how="left") \
                .merge(lv[keep].rename(columns={"id": "answer_id"}), on="answer_id", how="left")
    q = table["answer_id"].map(quality).fillna(table["duplicate_of"].map(quality))   # duplicates: the kept copy's value
    table.insert(table.columns.get_loc("speech_span_s"), "quality_composite", q)
    for c in ("hesitation_reasons", "expressiveness_reasons"):
        table[c] = table[c].fillna("")
    table.to_csv(out / "audio_answers.csv", index=False)

    ids = np.concatenate([rv["id"].to_numpy(str), mit["id"].to_numpy(str)])
    np.savez_compressed(out / "audio_embeddings_wavlm_l7.npz", answer_id=ids,
                        embedding=np.concatenate([X_rv, X_mit]).astype(np.float32))
    shutil.copyfile(P.EVIDENCE_V2, out / "evidence.jsonl")
    metrics.to_csv(out / "score_model_metrics.csv", index=False)

    miss = table[[f"pred_{t}" for t in TARGETS] + ["hesitation_index", "expressiveness_index"]].isna().sum()
    print(f"wrote {out}: {len(table)} answers ({(table['dataset'] == 'recruitview').sum()} RecruitView, "
          f"{(table['dataset'] == 'mit').sum()} MIT); embeddings {X_rv.shape[1]}-d")
    print("missing values:", miss[miss > 0].to_dict() or "none")


if __name__ == "__main__":
    main()
