# Repeated participant-grouped CV (Step A)

Protocol: 5 repeats × 5 outer folds, with 5-fold inner GroupKFold hyperparameter selection. Folds are grouped by `user_no`; every participant is validation exactly once per repeat. Train and validation participants are combined. The fixed test split is excluded from features, fitting, fold generation and scoring.

All preprocessing is fitted within each inner/outer training fold. SSL embeddings and handcrafted/eGeMAPS tables are loaded from existing caches; no feature extraction or test scoring is performed. The fixed test split is consulted only to verify participant disjointness. Metrics include pooled OOF predictions over all repeats and the mean/std of per-repeat metrics. Confidence intervals resample whole participants.

C-index is reported as `(Kendall tau-b + 1) / 2` to handle tied ordinal labels; it is a rank-concordance summary, not a survival-analysis estimate.

Participant bootstrap replicates: 1000. Models: Ridge, SVR, ExtraTrees. Ridge uses the stable LSQR solver. Inner hyperparameter selection minimizes mean grouped-fold RMSE.

## Pooled OOF metrics

| feature_set         | model      | target              | completed_repeats | pooled_mae | pooled_rmse | pooled_r2 | pooled_pearson | pooled_spearman | pooled_c_index | pearson_lower_95 | pearson_upper_95 | rmse_lower_95 | rmse_upper_95 |
| ------------------- | ---------- | ------------------- | ----------------- | ---------- | ----------- | --------- | -------------- | --------------- | -------------- | ---------------- | ---------------- | ------------- | ------------- |
| A_original          | ExtraTrees | confidence_score    | 5                 | 0.6763     | 1.0050      | 0.1751    | 0.4189         | 0.4505          | 0.6557         | 0.3679           | 0.4662           | 0.9213        | 1.0948        |
| A_original          | ExtraTrees | overall_performance | 5                 | 0.7135     | 1.1035      | 0.1915    | 0.4377         | 0.5022          | 0.6736         | 0.3926           | 0.4813           | 0.9994        | 1.2061        |
| A_original          | ExtraTrees | speaking_skills     | 5                 | 0.7523     | 1.1805      | 0.1584    | 0.3981         | 0.4524          | 0.6563         | 0.3548           | 0.4384           | 1.0645        | 1.2983        |
| A_original          | Ridge      | confidence_score    | 5                 | 0.6816     | 1.0046      | 0.1758    | 0.4194         | 0.4458          | 0.6539         | 0.3677           | 0.4685           | 0.9207        | 1.0968        |
| A_original          | Ridge      | overall_performance | 5                 | 0.7181     | 1.1026      | 0.1927    | 0.4391         | 0.5031          | 0.6739         | 0.3904           | 0.4855           | 1.0025        | 1.2025        |
| A_original          | Ridge      | speaking_skills     | 5                 | 0.7554     | 1.1738      | 0.1678    | 0.4103         | 0.4543          | 0.6568         | 0.3620           | 0.4612           | 1.0677        | 1.2805        |
| A_original          | SVR        | confidence_score    | 5                 | 0.6776     | 1.0059      | 0.1736    | 0.4175         | 0.4489          | 0.6551         | 0.3670           | 0.4627           | 0.9232        | 1.0948        |
| A_original          | SVR        | overall_performance | 5                 | 0.7114     | 1.1058      | 0.1881    | 0.4363         | 0.5057          | 0.6750         | 0.3919           | 0.4807           | 1.0051        | 1.2060        |
| A_original          | SVR        | speaking_skills     | 5                 | 0.7471     | 1.1764      | 0.1641    | 0.4096         | 0.4541          | 0.6569         | 0.3638           | 0.4517           | 1.0670        | 1.2954        |
| C_original_plus_new | ExtraTrees | confidence_score    | 5                 | 0.6710     | 1.0021      | 0.1799    | 0.4252         | 0.4661          | 0.6614         | 0.3786           | 0.4734           | 0.9154        | 1.0927        |
| C_original_plus_new | ExtraTrees | overall_performance | 5                 | 0.7124     | 1.1048      | 0.1896    | 0.4357         | 0.5063          | 0.6751         | 0.3899           | 0.4789           | 0.9957        | 1.2000        |
| C_original_plus_new | ExtraTrees | speaking_skills     | 5                 | 0.7459     | 1.1755      | 0.1654    | 0.4074         | 0.4666          | 0.6612         | 0.3661           | 0.4515           | 1.0563        | 1.2829        |
| C_original_plus_new | Ridge      | confidence_score    | 5                 | 0.6777     | 1.0057      | 0.1740    | 0.4178         | 0.4431          | 0.6533         | 0.3642           | 0.4662           | 0.9218        | 1.0955        |
| C_original_plus_new | Ridge      | overall_performance | 5                 | 0.7192     | 1.1077      | 0.1854    | 0.4319         | 0.4941          | 0.6707         | 0.3842           | 0.4788           | 1.0008        | 1.2078        |
| C_original_plus_new | Ridge      | speaking_skills     | 5                 | 0.7564     | 1.1909      | 0.1434    | 0.3787         | 0.4411          | 0.6518         | 0.3281           | 0.4300           | 1.0787        | 1.2979        |
| C_original_plus_new | SVR        | confidence_score    | 5                 | 0.6743     | 1.0047      | 0.1756    | 0.4199         | 0.4581          | 0.6582         | 0.3724           | 0.4645           | 0.9148        | 1.0949        |
| C_original_plus_new | SVR        | overall_performance | 5                 | 0.7180     | 1.1122      | 0.1787    | 0.4254         | 0.4981          | 0.6719         | 0.3792           | 0.4739           | 1.0038        | 1.2082        |
| C_original_plus_new | SVR        | speaking_skills     | 5                 | 0.7510     | 1.1827      | 0.1551    | 0.3963         | 0.4480          | 0.6540         | 0.3496           | 0.4389           | 1.0629        | 1.3015        |
| HuBERT              | ExtraTrees | confidence_score    | 5                 | 0.6652     | 0.9944      | 0.1924    | 0.4392         | 0.4866          | 0.6690         | 0.3912           | 0.4845           | 0.9044        | 1.0836        |
| HuBERT              | ExtraTrees | overall_performance | 5                 | 0.7060     | 1.1080      | 0.1849    | 0.4304         | 0.5133          | 0.6778         | 0.3866           | 0.4741           | 1.0059        | 1.2131        |
| HuBERT              | ExtraTrees | speaking_skills     | 5                 | 0.7394     | 1.1763      | 0.1643    | 0.4058         | 0.4761          | 0.6644         | 0.3597           | 0.4482           | 1.0599        | 1.2890        |
| HuBERT              | Ridge      | confidence_score    | 5                 | 0.6607     | 0.9898      | 0.1998    | 0.4482         | 0.4961          | 0.6729         | 0.3983           | 0.4889           | 0.9050        | 1.0864        |
| HuBERT              | Ridge      | overall_performance | 5                 | 0.7085     | 1.1075      | 0.1856    | 0.4315         | 0.5140          | 0.6780         | 0.3863           | 0.4825           | 1.0069        | 1.2071        |
| HuBERT              | Ridge      | speaking_skills     | 5                 | 0.7383     | 1.1725      | 0.1697    | 0.4123         | 0.4820          | 0.6667         | 0.3638           | 0.4591           | 1.0572        | 1.2914        |
| HuBERT              | SVR        | confidence_score    | 5                 | 0.6668     | 0.9990      | 0.1849    | 0.4321         | 0.4770          | 0.6662         | 0.3843           | 0.4802           | 0.9030        | 1.0858        |
| HuBERT              | SVR        | overall_performance | 5                 | 0.7102     | 1.1069      | 0.1865    | 0.4340         | 0.5063          | 0.6760         | 0.3818           | 0.4796           | 1.0002        | 1.2121        |
| HuBERT              | SVR        | speaking_skills     | 5                 | 0.7413     | 1.1775      | 0.1625    | 0.4051         | 0.4750          | 0.6643         | 0.3629           | 0.4453           | 1.0560        | 1.2883        |
| WavLM               | ExtraTrees | confidence_score    | 5                 | 0.6675     | 1.0011      | 0.1814    | 0.4264         | 0.4766          | 0.6650         | 0.3771           | 0.4736           | 0.9122        | 1.0870        |
| WavLM               | ExtraTrees | overall_performance | 5                 | 0.7088     | 1.1074      | 0.1857    | 0.4316         | 0.5183          | 0.6791         | 0.3865           | 0.4754           | 0.9999        | 1.2121        |
| WavLM               | ExtraTrees | speaking_skills     | 5                 | 0.7394     | 1.1778      | 0.1621    | 0.4032         | 0.4767          | 0.6643         | 0.3534           | 0.4503           | 1.0584        | 1.3003        |
| WavLM               | Ridge      | confidence_score    | 5                 | 0.6593     | 0.9871      | 0.2043    | 0.4532         | 0.4989          | 0.6734         | 0.4009           | 0.5013           | 0.8989        | 1.0785        |
| WavLM               | Ridge      | overall_performance | 5                 | 0.7055     | 1.0990      | 0.1981    | 0.4459         | 0.5258          | 0.6819         | 0.3996           | 0.4879           | 1.0000        | 1.2013        |
| WavLM               | Ridge      | speaking_skills     | 5                 | 0.7312     | 1.1599      | 0.1874    | 0.4348         | 0.4929          | 0.6704         | 0.3900           | 0.4767           | 1.0514        | 1.2738        |
| WavLM               | SVR        | confidence_score    | 5                 | 0.6680     | 0.9959      | 0.1899    | 0.4377         | 0.4821          | 0.6674         | 0.3858           | 0.4836           | 0.8993        | 1.0827        |
| WavLM               | SVR        | overall_performance | 5                 | 0.7046     | 1.1022      | 0.1934    | 0.4431         | 0.5206          | 0.6808         | 0.3964           | 0.4860           | 0.9948        | 1.2140        |
| WavLM               | SVR        | speaking_skills     | 5                 | 0.7463     | 1.1789      | 0.1605    | 0.4016         | 0.4702          | 0.6620         | 0.3610           | 0.4399           | 1.0663        | 1.2939        |
| eGeMAPS             | ExtraTrees | confidence_score    | 5                 | 0.6947     | 1.0266      | 0.1392    | 0.3734         | 0.3950          | 0.6356         | 0.3147           | 0.4245           | 0.9328        | 1.1144        |
| eGeMAPS             | ExtraTrees | overall_performance | 5                 | 0.7444     | 1.1496      | 0.1226    | 0.3502         | 0.4134          | 0.6415         | 0.2953           | 0.4010           | 1.0378        | 1.2613        |
| eGeMAPS             | ExtraTrees | speaking_skills     | 5                 | 0.7866     | 1.2174      | 0.1048    | 0.3239         | 0.3622          | 0.6228         | 0.2711           | 0.3713           | 1.0910        | 1.3416        |
| eGeMAPS             | Ridge      | confidence_score    | 5                 | 0.7022     | 1.0313      | 0.1313    | 0.3625         | 0.3977          | 0.6362         | 0.3142           | 0.4106           | 0.9455        | 1.1146        |
| eGeMAPS             | Ridge      | overall_performance | 5                 | 0.7538     | 1.1580      | 0.1096    | 0.3313         | 0.3986          | 0.6359         | 0.2783           | 0.3816           | 1.0548        | 1.2646        |
| eGeMAPS             | Ridge      | speaking_skills     | 5                 | 0.7855     | 1.2231      | 0.0964    | 0.3108         | 0.3605          | 0.6224         | 0.2589           | 0.3587           | 1.1015        | 1.3301        |
| eGeMAPS             | SVR        | confidence_score    | 5                 | 0.6986     | 1.0299      | 0.1336    | 0.3660         | 0.3918          | 0.6340         | 0.3186           | 0.4094           | 0.9357        | 1.1223        |
| eGeMAPS             | SVR        | overall_performance | 5                 | 0.7456     | 1.1497      | 0.1224    | 0.3502         | 0.4119          | 0.6403         | 0.3016           | 0.3952           | 1.0384        | 1.2616        |
| eGeMAPS             | SVR        | speaking_skills     | 5                 | 0.7824     | 1.2256      | 0.0928    | 0.3054         | 0.3633          | 0.6232         | 0.2541           | 0.3557           | 1.1140        | 1.3387        |

## Paired comparison against original handcrafted A

Deltas are candidate minus A_original; negative RMSE or positive Pearson favors the candidate. The pre-registered practical thresholds are an RMSE reduction of at least 0.01 or Pearson increase of at least 0.015, with the paired participant-bootstrap interval excluding zero, on at least two targets. These comparisons are not corrected for multiple testing.

| target              | model      | feature_set         | reference  | n_participants | rmse_delta | rmse_delta_lower_95 | rmse_delta_upper_95 | pearson_delta | pearson_delta_lower_95 | pearson_delta_upper_95 | paired_rmse_reference | paired_rmse_candidate |
| ------------------- | ---------- | ------------------- | ---------- | -------------- | ---------- | ------------------- | ------------------- | ------------- | ---------------------- | ---------------------- | --------------------- | --------------------- |
| confidence_score    | ExtraTrees | C_original_plus_new | A_original | 280            | -0.0029    | -0.0090             | 0.0035              | 0.0062        | -0.0072                | 0.0192                 | 1.0050                | 1.0021                |
| confidence_score    | ExtraTrees | HuBERT              | A_original | 280            | -0.0106    | -0.0246             | 0.0035              | 0.0203        | -0.0074                | 0.0506                 | 1.0050                | 0.9944                |
| confidence_score    | ExtraTrees | WavLM               | A_original | 280            | -0.0039    | -0.0170             | 0.0088              | 0.0075        | -0.0171                | 0.0353                 | 1.0050                | 1.0011                |
| confidence_score    | ExtraTrees | eGeMAPS             | A_original | 280            | 0.0217     | 0.0047              | 0.0387              | -0.0455       | -0.0854                | -0.0107                | 1.0050                | 1.0266                |
| confidence_score    | Ridge      | C_original_plus_new | A_original | 280            | 0.0011     | -0.0073             | 0.0098              | -0.0017       | -0.0185                | 0.0138                 | 1.0046                | 1.0057                |
| confidence_score    | Ridge      | HuBERT              | A_original | 280            | -0.0147    | -0.0303             | 0.0011              | 0.0288        | -0.0016                | 0.0595                 | 1.0046                | 0.9898                |
| confidence_score    | Ridge      | WavLM               | A_original | 280            | -0.0175    | -0.0317             | -0.0028             | 0.0338        | 0.0058                 | 0.0611                 | 1.0046                | 0.9871                |
| confidence_score    | Ridge      | eGeMAPS             | A_original | 280            | 0.0268     | 0.0062              | 0.0478              | -0.0569       | -0.0980                | -0.0156                | 1.0046                | 1.0313                |
| confidence_score    | SVR        | C_original_plus_new | A_original | 280            | -0.0012    | -0.0102             | 0.0083              | 0.0024        | -0.0174                | 0.0209                 | 1.0059                | 1.0047                |
| confidence_score    | SVR        | HuBERT              | A_original | 280            | -0.0069    | -0.0215             | 0.0085              | 0.0146        | -0.0167                | 0.0445                 | 1.0059                | 0.9990                |
| confidence_score    | SVR        | WavLM               | A_original | 280            | -0.0099    | -0.0226             | 0.0029              | 0.0202        | -0.0067                | 0.0469                 | 1.0059                | 0.9959                |
| confidence_score    | SVR        | eGeMAPS             | A_original | 280            | 0.0241     | 0.0084              | 0.0415              | -0.0515       | -0.0889                | -0.0175                | 1.0059                | 1.0299                |
| overall_performance | ExtraTrees | C_original_plus_new | A_original | 280            | 0.0013     | -0.0036             | 0.0059              | -0.0020       | -0.0100                | 0.0062                 | 1.1035                | 1.1048                |
| overall_performance | ExtraTrees | HuBERT              | A_original | 280            | 0.0045     | -0.0094             | 0.0167              | -0.0073       | -0.0284                | 0.0172                 | 1.1035                | 1.1080                |
| overall_performance | ExtraTrees | WavLM               | A_original | 280            | 0.0040     | -0.0093             | 0.0160              | -0.0062       | -0.0273                | 0.0171                 | 1.1035                | 1.1074                |
| overall_performance | ExtraTrees | eGeMAPS             | A_original | 280            | 0.0461     | 0.0273              | 0.0639              | -0.0875       | -0.1229                | -0.0526                | 1.1035                | 1.1496                |
| overall_performance | Ridge      | C_original_plus_new | A_original | 280            | 0.0050     | -0.0016             | 0.0110              | -0.0072       | -0.0167                | 0.0024                 | 1.1026                | 1.1077                |
| overall_performance | Ridge      | HuBERT              | A_original | 280            | 0.0049     | -0.0124             | 0.0212              | -0.0076       | -0.0351                | 0.0210                 | 1.1026                | 1.1075                |
| overall_performance | Ridge      | WavLM               | A_original | 280            | -0.0037    | -0.0186             | 0.0123              | 0.0068        | -0.0204                | 0.0318                 | 1.1026                | 1.0990                |
| overall_performance | Ridge      | eGeMAPS             | A_original | 280            | 0.0554     | 0.0353              | 0.0755              | -0.1078       | -0.1444                | -0.0727                | 1.1026                | 1.1580                |
| overall_performance | SVR        | C_original_plus_new | A_original | 280            | 0.0064     | 0.0005              | 0.0117              | -0.0109       | -0.0211                | -0.0000                | 1.1058                | 1.1122                |
| overall_performance | SVR        | HuBERT              | A_original | 280            | 0.0011     | -0.0127             | 0.0144              | -0.0022       | -0.0270                | 0.0224                 | 1.1058                | 1.1069                |
| overall_performance | SVR        | WavLM               | A_original | 280            | -0.0036    | -0.0161             | 0.0077              | 0.0068        | -0.0144                | 0.0302                 | 1.1058                | 1.1022                |
| overall_performance | SVR        | eGeMAPS             | A_original | 280            | 0.0439     | 0.0265              | 0.0612              | -0.0861       | -0.1221                | -0.0508                | 1.1058                | 1.1497                |
| speaking_skills     | ExtraTrees | C_original_plus_new | A_original | 280            | -0.0049    | -0.0096             | -0.0005             | 0.0093        | 0.0010                 | 0.0176                 | 1.1805                | 1.1755                |
| speaking_skills     | ExtraTrees | HuBERT              | A_original | 280            | -0.0042    | -0.0174             | 0.0095              | 0.0077        | -0.0174                | 0.0318                 | 1.1805                | 1.1763                |
| speaking_skills     | ExtraTrees | WavLM               | A_original | 280            | -0.0026    | -0.0164             | 0.0097              | 0.0051        | -0.0173                | 0.0302                 | 1.1805                | 1.1778                |
| speaking_skills     | ExtraTrees | eGeMAPS             | A_original | 280            | 0.0370     | 0.0187              | 0.0546              | -0.0742       | -0.1119                | -0.0371                | 1.1805                | 1.2174                |
| speaking_skills     | Ridge      | C_original_plus_new | A_original | 280            | 0.0171     | 0.0005              | 0.0380              | -0.0316       | -0.0652                | -0.0015                | 1.1738                | 1.1909                |
| speaking_skills     | Ridge      | HuBERT              | A_original | 280            | -0.0013    | -0.0184             | 0.0167              | 0.0020        | -0.0278                | 0.0303                 | 1.1738                | 1.1725                |
| speaking_skills     | Ridge      | WavLM               | A_original | 280            | -0.0139    | -0.0310             | 0.0029              | 0.0244        | -0.0030                | 0.0513                 | 1.1738                | 1.1599                |
| speaking_skills     | Ridge      | eGeMAPS             | A_original | 280            | 0.0493     | 0.0263              | 0.0736              | -0.0996       | -0.1424                | -0.0616                | 1.1738                | 1.2231                |
| speaking_skills     | SVR        | C_original_plus_new | A_original | 280            | 0.0063     | -0.0005             | 0.0132              | -0.0133       | -0.0268                | 0.0000                 | 1.1764                | 1.1827                |
| speaking_skills     | SVR        | HuBERT              | A_original | 280            | 0.0011     | -0.0111             | 0.0135              | -0.0046       | -0.0284                | 0.0189                 | 1.1764                | 1.1775                |
| speaking_skills     | SVR        | WavLM               | A_original | 280            | 0.0025     | -0.0086             | 0.0147              | -0.0081       | -0.0308                | 0.0144                 | 1.1764                | 1.1789                |
| speaking_skills     | SVR        | eGeMAPS             | A_original | 280            | 0.0492     | 0.0307              | 0.0672              | -0.1042       | -0.1470                | -0.0629                | 1.1764                | 1.2256                |

- C_original_plus_new / ExtraTrees: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- C_original_plus_new / Ridge: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- C_original_plus_new / SVR: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- HuBERT / ExtraTrees: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- HuBERT / Ridge: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- HuBERT / SVR: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- WavLM / ExtraTrees: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- WavLM / Ridge: 1/3 targets meet at least one threshold (confidence_score; RMSE threshold hits=1, Pearson threshold hits=1).
- WavLM / SVR: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- eGeMAPS / ExtraTrees: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- eGeMAPS / Ridge: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).
- eGeMAPS / SVR: 0/3 targets meet at least one threshold (none; RMSE threshold hits=0, Pearson threshold hits=0).

## Run status

Outer tasks completed: 1125; failed: 0.

## Files

- `cv_splits.json`: participant assignments for each outer repeat/fold.
- `oof_predictions.csv`: per-clip outer-fold predictions and selected hyperparameters.
- `repeated_cv_metrics.csv`: pooled metrics, repeat summaries, and bootstrap intervals.
- `paired_bootstrap_differences.csv`: candidate-vs-original participant-paired deltas.
- `fold_status.csv` and `checkpoint.json`: run progress and resume state.
