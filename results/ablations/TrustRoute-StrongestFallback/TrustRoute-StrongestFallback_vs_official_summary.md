# TrustRoute-StrongestFallback vs Official

| Dataset | Official Acc | Official Cost | Official Lat | Variant Acc | Variant Cost | Variant Lat | Delta Acc | Delta Cost | Delta Lat | Outcome |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| arc_challenge | 0.9706 | 2.2594e-04 | 2.111 | 0.9267 | 2.1354e-05 | 1.569 | -0.0439 | -2.0458e-04 | -0.543 | official_acc_better_tradeoff_mixed |
| hellaswag | 0.8522 | 2.3457e-03 | 11.802 | 0.8000 | 1.8206e-03 | 9.669 | -0.0522 | -5.2511e-04 | -2.133 | official_acc_better_tradeoff_mixed |
| humaneval | 0.9775 | 2.2730e-03 | 8.239 | 0.8941 | 9.5702e-05 | 2.662 | -0.0833 | -2.1773e-03 | -5.577 | official_acc_better_tradeoff_mixed |
| mbpp | 0.9247 | 2.5807e-03 | 8.184 | 0.9247 | 2.5352e-03 | 11.442 | +0.0000 | -4.5471e-05 | +3.258 | variant_acc_better_tradeoff_mixed |
| mmlu | 0.8959 | 1.7439e-03 | 4.374 | 0.7437 | 1.7800e-04 | 1.991 | -0.1522 | -1.5659e-03 | -2.383 | official_acc_better_tradeoff_mixed |
| winogrande | 0.8878 | 9.3571e-04 | 4.283 | 0.7131 | 2.9067e-05 | 1.834 | -0.1747 | -9.0664e-04 | -2.449 | official_acc_better_tradeoff_mixed |
