# Results

Two families of results, kept apart:

| Folder | What it answers | Produced by |
|---|---|---|
| [`audio_score_models/`](audio_score_models/README.md) | How well can audio predict the RecruitView scores (confidence, speaking skills, overall)? Feature/encoder/layer comparisons and the repeated grouped CV benchmark. | `scripts/legacy/` |
| [`explainability/`](explainability/README.md) | *Why* — which measurable delivery behaviours (pace, pauses, length, pitch...) relate to the scores, and the evidence that feeds the per-answer feedback. Stages 0–7 of `markdown/EXPLAINABILITY_STAGES.md`. | `scripts/explainability/` |

Not in this folder (gitignored, under `outputs/`): feature/embedding caches, the GRU and attention experiments
(`outputs/audio/reports/wavlm_gru_*`, `wavlm_attention_experiments/`), and the WavLM out-of-fold predictions
recovered from git for Stage 3 (`outputs/explainability/`).

**Where to start reading:** `explainability/README.md` (one paragraph per stage), then
`explainability/stage3_recruitview_validation/notes.md` for the current conclusions.
