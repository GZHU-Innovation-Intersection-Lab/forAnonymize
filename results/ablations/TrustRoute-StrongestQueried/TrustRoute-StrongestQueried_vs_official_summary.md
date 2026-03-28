# TrustRoute-StrongestQueried vs Official

| Dataset | Official Acc | Official Cost | Official Lat | Variant Acc | Variant Cost | Variant Lat | Delta Acc | Delta Cost | Delta Lat | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| arc_challenge | 0.9706 | 2.2594e-04 | 2.111 | 0.9188 | 2.1366e-05 | 1.593 | -0.0518 | -2.0457e-04 | -0.519 | official_acc_better_tradeoff_mixed |
| hellaswag | 0.8522 | 2.3457e-03 | 11.802 | 0.8111 | 2.3737e-03 | 20.121 | -0.0411 | +2.8052e-05 | +8.319 | official_3d_dominates_variant |
| humaneval | 0.9775 | 2.2730e-03 | 8.239 | 0.9324 | 4.7062e-04 | 14.108 | -0.0450 | -1.8024e-03 | +5.869 | official_acc_better_tradeoff_mixed |
| mbpp | 0.9247 | 2.5807e-03 | 8.184 | 0.9238 | 1.7946e-03 | 29.799 | -0.0010 | -7.8605e-04 | +21.614 | official_acc_better_tradeoff_mixed |
| mmlu | 0.8959 | 1.7439e-03 | 4.374 | 0.8163 | 2.5131e-04 | 6.838 | -0.0796 | -1.4926e-03 | +2.464 | official_acc_better_tradeoff_mixed |
| winogrande | 0.8878 | 9.3571e-04 | 4.283 | 0.7157 | 3.0402e-05 | 1.922 | -0.1721 | -9.0531e-04 | -2.361 | official_acc_better_tradeoff_mixed |
