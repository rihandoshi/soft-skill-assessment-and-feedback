# VAD and pause pooling diagnostic

Validation uses the existing participant-disjoint split. Ridge alpha was selected using training-only GroupKFold by participant (`user_id`); test participants were not scored. Each encoder/target uses its hidden layer selected by the prior train-only SSL sweep. Full-audio rows reproduce the existing baseline.

Silero VAD regions measure speech and non-speech acoustically. Leading/trailing non-speech contributes to total non-speech duration and `pause_ratio`; pause counts/duration summaries use internal gaps between speech regions. Clips with no detected speech use a zero speech embedding and have zero speech duration.

## Validation metrics

| encoder | target | method | mae | rmse | r2 | pearson | spearman |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wavlm | confidence_score | full_audio_mean_std | 0.6494 | 0.9566 | 0.1345 | 0.3695 | 0.3391 |
| wavlm | confidence_score | speech_only_mean_std | 0.6591 | 0.9645 | 0.1202 | 0.3524 | 0.3261 |
| wavlm | confidence_score | speech_only_plus_pause | 0.6578 | 0.9625 | 0.1239 | 0.3570 | 0.3305 |
| wavlm | confidence_score | speech_only_plus_duration | 0.6586 | 0.9634 | 0.1222 | 0.3549 | 0.3287 |
| wavlm | speaking_skills | full_audio_mean_std | 0.7013 | 1.0753 | 0.1515 | 0.3894 | 0.4216 |
| wavlm | speaking_skills | speech_only_mean_std | 0.7115 | 1.0900 | 0.1282 | 0.3601 | 0.3912 |
| wavlm | speaking_skills | speech_only_plus_pause | 0.7097 | 1.0881 | 0.1312 | 0.3640 | 0.3981 |
| wavlm | speaking_skills | speech_only_plus_duration | 0.7104 | 1.0888 | 0.1301 | 0.3626 | 0.3961 |
| wavlm | overall_performance | full_audio_mean_std | 0.6776 | 1.0659 | 0.1158 | 0.3479 | 0.4129 |
| wavlm | overall_performance | speech_only_mean_std | 0.6881 | 1.0752 | 0.1004 | 0.3277 | 0.3963 |
| wavlm | overall_performance | speech_only_plus_pause | 0.6854 | 1.0712 | 0.1070 | 0.3360 | 0.4058 |
| wavlm | overall_performance | speech_only_plus_duration | 0.6863 | 1.0728 | 0.1043 | 0.3325 | 0.4014 |
| hubert | confidence_score | full_audio_mean_std | 0.6404 | 0.9568 | 0.1342 | 0.3666 | 0.3447 |
| hubert | confidence_score | speech_only_mean_std | 0.6433 | 0.9683 | 0.1133 | 0.3378 | 0.3412 |
| hubert | confidence_score | speech_only_plus_pause | 0.6411 | 0.9651 | 0.1190 | 0.3458 | 0.3481 |
| hubert | confidence_score | speech_only_plus_duration | 0.6424 | 0.9664 | 0.1167 | 0.3425 | 0.3448 |
| hubert | speaking_skills | full_audio_mean_std | 0.6952 | 1.0757 | 0.1508 | 0.3903 | 0.4410 |
| hubert | speaking_skills | speech_only_mean_std | 0.7022 | 1.0876 | 0.1319 | 0.3639 | 0.4192 |
| hubert | speaking_skills | speech_only_plus_pause | 0.6997 | 1.0851 | 0.1360 | 0.3696 | 0.4264 |
| hubert | speaking_skills | speech_only_plus_duration | 0.7009 | 1.0863 | 0.1340 | 0.3669 | 0.4232 |
| hubert | overall_performance | full_audio_mean_std | 0.6674 | 1.0648 | 0.1177 | 0.3452 | 0.4189 |
| hubert | overall_performance | speech_only_mean_std | 0.6738 | 1.0757 | 0.0996 | 0.3177 | 0.4024 |
| hubert | overall_performance | speech_only_plus_pause | 0.6693 | 1.0692 | 0.1105 | 0.3333 | 0.4169 |
| hubert | overall_performance | speech_only_plus_duration | 0.6712 | 1.0720 | 0.1058 | 0.3266 | 0.4125 |

## Paired participant bootstrap

RMSE deltas are method minus reference; negative values favor the method. Intervals resample whole validation participants.

| encoder | target | method | reference | delta | lower_95 | upper_95 |
| --- | --- | --- | --- | --- | --- | --- |
| hubert | confidence_score | speech_only_mean_std | full_audio_mean_std | 0.0115 | -0.0080 | 0.0324 |
| hubert | confidence_score | speech_only_plus_pause | speech_only_mean_std | -0.0031 | -0.0050 | -0.0010 |
| hubert | confidence_score | speech_only_plus_pause | speech_only_plus_duration | -0.0013 | -0.0021 | -0.0005 |
| hubert | overall_performance | speech_only_mean_std | full_audio_mean_std | 0.0109 | -0.0111 | 0.0342 |
| hubert | overall_performance | speech_only_plus_pause | speech_only_mean_std | -0.0065 | -0.0092 | -0.0036 |
| hubert | overall_performance | speech_only_plus_pause | speech_only_plus_duration | -0.0028 | -0.0042 | -0.0012 |
| hubert | speaking_skills | speech_only_mean_std | full_audio_mean_std | 0.0119 | 0.0003 | 0.0247 |
| hubert | speaking_skills | speech_only_plus_pause | speech_only_mean_std | -0.0026 | -0.0046 | -0.0008 |
| hubert | speaking_skills | speech_only_plus_pause | speech_only_plus_duration | -0.0013 | -0.0023 | -0.0002 |
| wavlm | confidence_score | speech_only_mean_std | full_audio_mean_std | 0.0079 | -0.0071 | 0.0266 |
| wavlm | confidence_score | speech_only_plus_pause | speech_only_mean_std | -0.0020 | -0.0033 | -0.0007 |
| wavlm | confidence_score | speech_only_plus_pause | speech_only_plus_duration | -0.0009 | -0.0014 | -0.0004 |
| wavlm | overall_performance | speech_only_mean_std | full_audio_mean_std | 0.0093 | -0.0054 | 0.0259 |
| wavlm | overall_performance | speech_only_plus_pause | speech_only_mean_std | -0.0040 | -0.0055 | -0.0024 |
| wavlm | overall_performance | speech_only_plus_pause | speech_only_plus_duration | -0.0016 | -0.0024 | -0.0007 |
| wavlm | speaking_skills | speech_only_mean_std | full_audio_mean_std | 0.0147 | 0.0012 | 0.0298 |
| wavlm | speaking_skills | speech_only_plus_pause | speech_only_mean_std | -0.0019 | -0.0038 | -0.0003 |
| wavlm | speaking_skills | speech_only_plus_pause | speech_only_plus_duration | -0.0007 | -0.0014 | -0.0001 |

## Pause association after duration control

Associations below use training data only. Partial Spearman is computed on rank residuals after controlling for duration.

| target | feature | spearman_train | partial_spearman_controlling_duration_train |
| --- | --- | --- | --- |
| confidence_score | duration_seconds | -0.353 | nan |
| confidence_score | pause_ratio | 0.354 | 0.265 |
| confidence_score | internal_pause_ratio | 0.050 | 0.161 |
| confidence_score | number_of_pauses | -0.225 | 0.127 |
| confidence_score | mean_pause_duration_seconds | 0.019 | 0.113 |
| confidence_score | maximum_pause_duration_seconds | -0.038 | 0.160 |
| confidence_score | mean_rms | -0.372 | -0.294 |
| speaking_skills | duration_seconds | -0.380 | nan |
| speaking_skills | pause_ratio | 0.335 | 0.235 |
| speaking_skills | internal_pause_ratio | 0.031 | 0.149 |
| speaking_skills | number_of_pauses | -0.248 | 0.128 |
| speaking_skills | mean_pause_duration_seconds | -0.002 | 0.097 |
| speaking_skills | maximum_pause_duration_seconds | -0.062 | 0.149 |
| speaking_skills | mean_rms | -0.333 | -0.243 |
| overall_performance | duration_seconds | -0.421 | nan |
| overall_performance | pause_ratio | 0.375 | 0.270 |
| overall_performance | internal_pause_ratio | 0.039 | 0.175 |
| overall_performance | number_of_pauses | -0.275 | 0.145 |
| overall_performance | mean_pause_duration_seconds | 0.004 | 0.118 |
| overall_performance | maximum_pause_duration_seconds | -0.070 | 0.166 |
| overall_performance | mean_rms | -0.376 | -0.284 |

## Interpretation

Compare `speech_only_plus_pause` with `speech_only_mean_std` to assess whether adding pauses helps. Compare `speech_only_plus_pause` with `speech_only_plus_duration` to assess whether pause summaries contribute beyond a duration control. Treat these as diagnostic comparisons; small point-estimate differences do not establish a general model winner.
