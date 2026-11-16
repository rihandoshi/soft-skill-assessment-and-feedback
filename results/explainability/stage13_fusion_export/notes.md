# Stage 13 notes: hand-over to fusion (Phase 6, 16 Nov 2026)

**Decision (16 Nov):**
- Speaker segmentation of new recordings (Phase 4, Test 2) is deferred.
- The current priority is giving the per-answer audio outputs to the teammate who builds the late-fusion model.
- The full pipeline will be demoed once, after fusion.

**What was done:** `export_for_fusion.py` (CPU, about 40 s) writes this folder. `SCHEMA.md` is the file for the fusion owner: columns, the leakage rule, and how to load it.

| Check | Result |
|---|---|
| Coverage | All 2,011 RecruitView answers in `Datasets/Split Dataset` and all 674 MIT answers |
| Keys | `person_id` = `user_no`; the splits match `Datasets/Split Dataset` exactly (231 / 49 / 51 people) |
| Score predictions | All 12 RecruitView targets: WavLM L7 + Ridge, out-of-fold on train (5-fold, grouped by person). Val Spearman 0.24–0.43, test 0.25–0.55 |
| Feedback dimensions | Hesitation and expressiveness index, level and components, from `answer_levels.csv` (Phase 2 settings). Missing for 46 answers with too little speech |

**Findings worth passing on** (details in `SCHEMA.md`):
1. **The 12 score predictions are one signal.** The first component holds 94% of their variance; the median correlation is 0.97.
2. **That signal follows speaking time** (ρ about −0.6).
3. **The feedback dimensions are not redundant with it.** Hesitation is nearly independent of it (ρ 0.02) and expressiveness runs against it (−0.39), so they add information.

**Not done in Phase 6:** the shared three-branch schema (README section 11, step 3). `SCHEMA.md` covers the audio side; the other branches have to agree on theirs.

**Re-run:** `python scripts/explainability/export_for_fusion.py`. Re-run it after any change to `fit_answer_levels.py` or `build_answer_evidence.py`.
