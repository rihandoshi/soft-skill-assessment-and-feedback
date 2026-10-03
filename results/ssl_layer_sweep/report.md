# WavLM/HuBERT layer and pooling diagnostic

All hidden-state outputs available in the existing cache were evaluated. Ridge alpha was selected from training-only GroupKFold folds grouped by participant, then the chosen configuration was evaluated on the fixed participant-disjoint validation split. Test rows were not used for model fitting or scoring.

Layer 0 is the feature-encoder output; layers 1–12 are Transformer block outputs. For each layer, mean, population standard deviation, and concatenated mean+std pooling were evaluated separately. Mean/std are over time frames after the existing chunk-overlap trimming and configured speech-pooling policy.

The weighted model is a nonnegative, sum-to-one blend of mean+std layer predictions. Its weights are learned by minimizing error against the training labels using only training participant-grouped OOF predictions. No validation information enters the weights.

## Comparison

See `comparison.csv` for MAE, RMSE, R², Pearson and Spearman; participant-cluster bootstrap intervals are included for R², Pearson and Spearman. `paired_rmse_differences.csv` reports participant-paired 95% bootstrap intervals: intervals crossing zero are inconclusive. Small point-estimate differences are not called wins.

| Encoder | Target | Method | Chosen layer | Pooling | MAE | RMSE | R² | Pearson | Spearman |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| wavlm | confidence_score | current_mean_std | layer_7 | mean+std | 0.6494 | 0.9566 | 0.1345 | 0.3695 | 0.3391 |
| wavlm | confidence_score | best_individual | layer_7 | mean+std | 0.6494 | 0.9566 | 0.1345 | 0.3695 | 0.3391 |
| wavlm | confidence_score | weighted_layers | weighted_all_layers | mean+std | 0.6426 | 0.9543 | 0.1386 | 0.3740 | 0.3553 |
| wavlm | speaking_skills | current_mean_std | layer_7 | mean+std | 0.7013 | 1.0753 | 0.1515 | 0.3894 | 0.4216 |
| wavlm | speaking_skills | best_individual | layer_7 | mean+std | 0.7013 | 1.0753 | 0.1515 | 0.3894 | 0.4216 |
| wavlm | speaking_skills | weighted_layers | weighted_all_layers | mean+std | 0.6995 | 1.0740 | 0.1535 | 0.3920 | 0.4250 |
| wavlm | overall_performance | current_mean_std | layer_7 | mean+std | 0.6776 | 1.0659 | 0.1158 | 0.3479 | 0.4129 |
| wavlm | overall_performance | best_individual | layer_7 | mean+std | 0.6776 | 1.0659 | 0.1158 | 0.3479 | 0.4129 |
| wavlm | overall_performance | weighted_layers | weighted_all_layers | mean+std | 0.6721 | 1.0619 | 0.1226 | 0.3557 | 0.4319 |
| hubert | confidence_score | current_mean_std | layer_6 | mean+std | 0.6404 | 0.9568 | 0.1342 | 0.3666 | 0.3447 |
| hubert | confidence_score | best_individual | layer_6 | mean+std | 0.6404 | 0.9568 | 0.1342 | 0.3666 | 0.3447 |
| hubert | confidence_score | weighted_layers | weighted_all_layers | mean+std | 0.6381 | 0.9558 | 0.1359 | 0.3688 | 0.3444 |
| hubert | speaking_skills | current_mean_std | layer_9 | mean+std | 0.6952 | 1.0757 | 0.1508 | 0.3903 | 0.4410 |
| hubert | speaking_skills | best_individual | layer_9 | mean+std | 0.6952 | 1.0757 | 0.1508 | 0.3903 | 0.4410 |
| hubert | speaking_skills | weighted_layers | weighted_all_layers | mean+std | 0.6920 | 1.0745 | 0.1528 | 0.3931 | 0.4402 |
| hubert | overall_performance | current_mean_std | layer_6 | mean+std | 0.6674 | 1.0648 | 0.1177 | 0.3452 | 0.4189 |
| hubert | overall_performance | best_individual | layer_6 | std | 0.6870 | 1.0714 | 0.1068 | 0.3438 | 0.4311 |
| hubert | overall_performance | weighted_layers | weighted_all_layers | mean+std | 0.6694 | 1.0621 | 0.1222 | 0.3506 | 0.4257 |

## Layer and pooling curves

`layer_experiments.csv` contains every evaluated layer × pooling × target × encoder, including grouped-CV fold metrics, validation metrics, selected alpha and whether the prior mean+std CV score was reused. Plots show validation R²/RMSE by layer for each pooling method.

## Weighted aggregation

`weighted_layer_weights.csv` lists target-specific weights learned only from training OOF predictions. Weights are constrained to be nonnegative and sum to one. Weighted rows do not report a non-nested CV estimate because fitting the blend weights on full-training OOF labels would make such an estimate optimistic; their held-out validation metrics remain independent of that fitting.

## Reproducibility and cache

`experiment_config.json` records model checkpoints, preprocessing, pooling definitions, layer indices, alpha grid, split/CV settings and source cache paths. All-layer means and standard deviations are persisted under the existing ignored audio cache tree with a preprocessing fingerprint. The experiment reused the existing per-split Transformer embeddings; no WavLM/HuBERT forward pass was required.

## Diagnostic interpretation

Use paired participant-bootstrap intervals to judge whether layer selection, weighted aggregation or encoder differences are distinguishable from zero. Cross-target consistency matters; a small improvement on one target without a corresponding interval-supported pattern is inconclusive. No model winner is asserted in this report.
