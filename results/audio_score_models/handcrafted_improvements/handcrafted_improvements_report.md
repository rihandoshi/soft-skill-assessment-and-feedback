# Handcrafted feature improvement comparison

All validation results use the existing participant-disjoint split. Hyperparameters were chosen by train-only `GroupKFold(user_id)`; the test split was not scored. Participant-cluster bootstrap intervals and paired RMSE deltas are saved separately.

## Feature sets

| set | dimensions |
| --- | --- |
| A_original | 27 |
| B_improved_only | 36 |
| C_original_plus_new | 63 |

## Validation results

| feature_set | model | target | mae | rmse | r2 | pearson | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_original | ExtraTrees | confidence_score | 0.6307 | 0.9461 | 0.1533 | 0.3917 | 0.3525 |
| B_improved_only | ExtraTrees | confidence_score | 0.6425 | 0.9580 | 0.1319 | 0.3707 | 0.3634 |
| C_original_plus_new | ExtraTrees | confidence_score | 0.6295 | 0.9490 | 0.1483 | 0.3859 | 0.3631 |
| A_original | RandomForest | confidence_score | 0.6384 | 0.9470 | 0.1519 | 0.3938 | 0.3559 |
| B_improved_only | RandomForest | confidence_score | 0.6389 | 0.9601 | 0.1282 | 0.3618 | 0.3574 |
| C_original_plus_new | RandomForest | confidence_score | 0.6366 | 0.9490 | 0.1483 | 0.3892 | 0.3655 |
| A_original | Ridge | confidence_score | 0.6397 | 0.9436 | 0.1579 | 0.4050 | 0.3629 |
| B_improved_only | Ridge | confidence_score | 0.6340 | 0.9605 | 0.1275 | 0.3602 | 0.3542 |
| C_original_plus_new | Ridge | confidence_score | 0.6255 | 0.9453 | 0.1549 | 0.3935 | 0.3511 |
| A_original | SVR | confidence_score | 0.6320 | 0.9431 | 0.1587 | 0.3990 | 0.3585 |
| B_improved_only | SVR | confidence_score | 0.6434 | 0.9611 | 0.1263 | 0.3621 | 0.3358 |
| C_original_plus_new | SVR | confidence_score | 0.6465 | 0.9566 | 0.1344 | 0.3722 | 0.3533 |
| A_original | ExtraTrees | overall_performance | 0.6496 | 1.0266 | 0.1799 | 0.4260 | 0.5025 |
| B_improved_only | ExtraTrees | overall_performance | 0.6512 | 1.0401 | 0.1581 | 0.3984 | 0.4879 |
| C_original_plus_new | ExtraTrees | overall_performance | 0.6475 | 1.0337 | 0.1685 | 0.4107 | 0.4885 |
| A_original | RandomForest | overall_performance | 0.6605 | 1.0380 | 0.1616 | 0.4103 | 0.4874 |
| B_improved_only | RandomForest | overall_performance | 0.6641 | 1.0499 | 0.1423 | 0.3883 | 0.4685 |
| C_original_plus_new | RandomForest | overall_performance | 0.6582 | 1.0391 | 0.1597 | 0.4081 | 0.4947 |
| A_original | Ridge | overall_performance | 0.6385 | 1.0185 | 0.1929 | 0.4416 | 0.5020 |
| B_improved_only | Ridge | overall_performance | 0.6578 | 1.0430 | 0.1535 | 0.3927 | 0.4631 |
| C_original_plus_new | Ridge | overall_performance | 0.6363 | 1.0235 | 0.1849 | 0.4313 | 0.4894 |
| A_original | SVR | overall_performance | 0.6381 | 1.0228 | 0.1859 | 0.4384 | 0.4981 |
| B_improved_only | SVR | overall_performance | 0.6750 | 1.0592 | 0.1269 | 0.3598 | 0.4307 |
| C_original_plus_new | SVR | overall_performance | 0.6362 | 1.0251 | 0.1823 | 0.4318 | 0.4950 |
| A_original | ExtraTrees | speaking_skills | 0.6814 | 1.0735 | 0.1543 | 0.3948 | 0.4700 |
| B_improved_only | ExtraTrees | speaking_skills | 0.7062 | 1.0787 | 0.1461 | 0.3859 | 0.4565 |
| C_original_plus_new | ExtraTrees | speaking_skills | 0.6868 | 1.0721 | 0.1566 | 0.3968 | 0.4687 |
| A_original | RandomForest | speaking_skills | 0.7058 | 1.0895 | 0.1289 | 0.3676 | 0.4334 |
| B_improved_only | RandomForest | speaking_skills | 0.7016 | 1.0797 | 0.1445 | 0.3813 | 0.4489 |
| C_original_plus_new | RandomForest | speaking_skills | 0.6980 | 1.0792 | 0.1453 | 0.3852 | 0.4540 |
| A_original | Ridge | speaking_skills | 0.7036 | 1.0925 | 0.1241 | 0.3675 | 0.4288 |
| B_improved_only | Ridge | speaking_skills | 0.6904 | 1.0883 | 0.1308 | 0.3675 | 0.4515 |
| C_original_plus_new | Ridge | speaking_skills | 0.6873 | 1.0826 | 0.1400 | 0.3749 | 0.4369 |
| A_original | SVR | speaking_skills | 0.6940 | 1.0859 | 0.1347 | 0.3673 | 0.4400 |
| B_improved_only | SVR | speaking_skills | 0.7114 | 1.0941 | 0.1216 | 0.3507 | 0.4176 |
| C_original_plus_new | SVR | speaking_skills | 0.7144 | 1.1046 | 0.1046 | 0.3265 | 0.4012 |

## Paired RMSE deltas

Delta is feature-set RMSE minus reference RMSE; negative favors the first set. Confidence intervals resample participants.

| target | model | feature_set | reference | rmse_delta | lower_95 | upper_95 | bootstrap_unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confidence_score | ExtraTrees | B_improved_only | A_original | 0.0119 | -0.0063 | 0.0326 | participant |
| confidence_score | ExtraTrees | C_original_plus_new | A_original | 0.0028 | -0.0078 | 0.0136 | participant |
| confidence_score | ExtraTrees | C_original_plus_new | B_improved_only | -0.0091 | -0.0243 | 0.0033 | participant |
| confidence_score | RandomForest | B_improved_only | A_original | 0.0131 | -0.0116 | 0.0387 | participant |
| confidence_score | RandomForest | C_original_plus_new | A_original | 0.0020 | -0.0101 | 0.0144 | participant |
| confidence_score | RandomForest | C_original_plus_new | B_improved_only | -0.0111 | -0.0261 | 0.0036 | participant |
| confidence_score | Ridge | B_improved_only | A_original | 0.0168 | -0.0105 | 0.0420 | participant |
| confidence_score | Ridge | C_original_plus_new | A_original | 0.0017 | -0.0157 | 0.0176 | participant |
| confidence_score | Ridge | C_original_plus_new | B_improved_only | -0.0152 | -0.0293 | 0.0009 | participant |
| confidence_score | SVR | B_improved_only | A_original | 0.0180 | -0.0059 | 0.0437 | participant |
| confidence_score | SVR | C_original_plus_new | A_original | 0.0135 | -0.0064 | 0.0340 | participant |
| confidence_score | SVR | C_original_plus_new | B_improved_only | -0.0045 | -0.0206 | 0.0123 | participant |
| overall_performance | ExtraTrees | B_improved_only | A_original | 0.0135 | -0.0054 | 0.0335 | participant |
| overall_performance | ExtraTrees | C_original_plus_new | A_original | 0.0071 | -0.0044 | 0.0183 | participant |
| overall_performance | ExtraTrees | C_original_plus_new | B_improved_only | -0.0064 | -0.0199 | 0.0062 | participant |
| overall_performance | RandomForest | B_improved_only | A_original | 0.0119 | -0.0115 | 0.0362 | participant |
| overall_performance | RandomForest | C_original_plus_new | A_original | 0.0011 | -0.0115 | 0.0124 | participant |
| overall_performance | RandomForest | C_original_plus_new | B_improved_only | -0.0107 | -0.0307 | 0.0066 | participant |
| overall_performance | Ridge | B_improved_only | A_original | 0.0246 | -0.0003 | 0.0470 | participant |
| overall_performance | Ridge | C_original_plus_new | A_original | 0.0050 | -0.0062 | 0.0160 | participant |
| overall_performance | Ridge | C_original_plus_new | B_improved_only | -0.0195 | -0.0389 | -0.0022 | participant |
| overall_performance | SVR | B_improved_only | A_original | 0.0364 | -0.0009 | 0.0723 | participant |
| overall_performance | SVR | C_original_plus_new | A_original | 0.0022 | -0.0117 | 0.0149 | participant |
| overall_performance | SVR | C_original_plus_new | B_improved_only | -0.0342 | -0.0613 | -0.0083 | participant |
| speaking_skills | ExtraTrees | B_improved_only | A_original | 0.0052 | -0.0194 | 0.0336 | participant |
| speaking_skills | ExtraTrees | C_original_plus_new | A_original | -0.0014 | -0.0110 | 0.0088 | participant |
| speaking_skills | ExtraTrees | C_original_plus_new | B_improved_only | -0.0067 | -0.0283 | 0.0106 | participant |
| speaking_skills | RandomForest | B_improved_only | A_original | -0.0098 | -0.0352 | 0.0136 | participant |
| speaking_skills | RandomForest | C_original_plus_new | A_original | -0.0103 | -0.0258 | 0.0043 | participant |
| speaking_skills | RandomForest | C_original_plus_new | B_improved_only | -0.0005 | -0.0142 | 0.0133 | participant |
| speaking_skills | Ridge | B_improved_only | A_original | -0.0042 | -0.0390 | 0.0239 | participant |
| speaking_skills | Ridge | C_original_plus_new | A_original | -0.0100 | -0.0321 | 0.0068 | participant |
| speaking_skills | Ridge | C_original_plus_new | B_improved_only | -0.0058 | -0.0248 | 0.0133 | participant |
| speaking_skills | SVR | B_improved_only | A_original | 0.0082 | -0.0153 | 0.0328 | participant |
| speaking_skills | SVR | C_original_plus_new | A_original | 0.0187 | 0.0010 | 0.0424 | participant |
| speaking_skills | SVR | C_original_plus_new | B_improved_only | 0.0105 | -0.0140 | 0.0356 | participant |

## Feature definitions and limitations

New features include transcript word rates only when timed segments are present; acoustic pause count/ratio/duration summaries from internal Silero speech gaps; F0 level/range, voiced percentage, semitone-relative contour slopes; speech RMS level/variability/dynamic range/trend; and mean relative pitch, voiced percentage, and RMS over each answer third. No syllable counts are derived. Semitone normalization uses each recording’s median voiced F0, not participant identity or a cross-clip identity baseline.

All original baseline columns remain in A and C, including jitter/shimmer proxies. There was no HNR feature in the existing table. No existing features were dropped. New exports contain predictor values and clip IDs only; identifiers and target values are excluded from all model matrices.

Limitations: transcript times are coarse `[MM:SS - MM:SS]` segments and unavailable for some recordings; F0/autocorrelation can produce errors on noisy or low-volume speech; VAD can miss very quiet speech; jitter/shimmer remain uncalibrated proxies; the conclusions are limited to this validation cohort.
