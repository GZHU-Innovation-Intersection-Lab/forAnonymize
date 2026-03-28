# Runtime Method Audit

This file records the repository truth for the current `test90` submission line.
It exists to close reviewer questions about judge usage, routing state, proxy signals, and protocol truth.

## Bottom Line

- The active submission line is `7 benchmark + test90 + mixed-source TrustRoute`.
- The runtime entry point is [`run_exp.py#L275`](artifact/scripts/run_exp.py#L275) calling [`run_method#L42`](src/runner/executor.py#L42).
- `run_exp.py` passes `router=None` and `judge=None` into the dispatcher at [`run_exp.py#L277`](artifact/scripts/run_exp.py#L277).
- The dispatcher constructs a local `judge_fn`, but it is a constant stub returning `0.5` at [`executor.py#L93`](src/runner/executor.py#L93).
- `run_ours_lite` accepts `judge_fn` in its signature at [`ours_lite_v2.py#L187`](src/runner/ours_lite_v2.py#L187), but the function body never calls it.
- Current ranking uses static price metadata, static `quality`, and persisted reputation at [`lite_utils.py#L105`](src/runner/lite_utils.py#L105).
- There is no implemented path for observed cost/latency tracking or EMA/rho smoothing in the active runtime.
- Non-executable tasks therefore rely on heuristic confidence, majority vote, and reputation update, not an active judge oracle.

## Active TrustRoute Source Selection

| Dataset | Selected Source | Seeds | Raw Files |
|---|---|---:|---|
| arc_challenge | `real_eval_locked` | 3 | `omitted_raw_runs/final_experiments/selected/arc_challenge_TrustRoute_seed1.csv|omitted_raw_runs/final_experiments/selected/arc_challenge_TrustRoute_seed2.csv|omitted_raw_runs/final_experiments/selected/arc_challenge_TrustRoute_seed3.csv` |
| hellaswag | `best_cfg_full` | 3 | `omitted_raw_runs/test90_runs/hellaswag/hellaswag_TrustRoute_best_tau1_0.95_tau2_1.0_wq_0.25_wr_0.6_wc_0.15_mr_0.0_seed1.csv|omitted_raw_runs/test90_runs/hellaswag/hellaswag_TrustRoute_best_tau1_0.95_tau2_1.0_wq_0.25_wr_0.6_wc_0.15_mr_0.0_seed2.csv|omitted_raw_runs/test90_runs/hellaswag/hellaswag_TrustRoute_best_tau1_0.95_tau2_1.0_wq_0.25_wr_0.6_wc_0.15_mr_0.0_seed3.csv` |
| winogrande | `best_cfg_full` | 3 | `omitted_raw_runs/test90_runs/winogrande/winogrande_TrustRoute_best_tau1_0.95_tau2_1.0_wq_0.25_wr_0.25_wc_0.5_mr_0.0_seed1.csv|omitted_raw_runs/test90_runs/winogrande/winogrande_TrustRoute_best_tau1_0.95_tau2_1.0_wq_0.25_wr_0.25_wc_0.5_mr_0.0_seed2.csv|omitted_raw_runs/test90_runs/winogrande/winogrande_TrustRoute_best_tau1_0.95_tau2_1.0_wq_0.25_wr_0.25_wc_0.5_mr_0.0_seed3.csv` |
| mmlu | `real_eval_locked` | 3 | `omitted_raw_runs/final_experiments/selected/mmlu_TrustRoute_seed1.csv|omitted_raw_runs/final_experiments/selected/mmlu_TrustRoute_seed2.csv|omitted_raw_runs/final_experiments/selected/mmlu_TrustRoute_seed3.csv` |
| humaneval | `real_eval_locked` | 3 | `omitted_raw_runs/final_experiments/selected/humaneval_TrustRoute_seed1.csv|omitted_raw_runs/final_experiments/selected/humaneval_TrustRoute_seed2.csv|omitted_raw_runs/final_experiments/selected/humaneval_TrustRoute_seed3.csv` |
| mbpp | `paper_main_v1` | 3 | `omitted_raw_runs/final_experiments/paper_main_v1/mbpp/TrustRoute_t1_0.75_t2_0.5_wq_0.4_wr_0.3_wc_0.3_mr_0.0_seed1.csv|omitted_raw_runs/final_experiments/paper_main_v1/mbpp/TrustRoute_t1_0.75_t2_0.5_wq_0.4_wr_0.3_wc_0.3_mr_0.0_seed2.csv|omitted_raw_runs/final_experiments/paper_main_v1/mbpp/TrustRoute_t1_0.75_t2_0.5_wq_0.4_wr_0.3_wc_0.3_mr_0.0_seed3.csv` |
| gsm8k | `paper_main_v1` | 3 | `omitted_raw_runs/final_experiments/paper_main_v1/gsm8k/TrustRoute_t1_0.75_t2_0.5_wq_0.4_wr_0.3_wc_0.3_mr_0.0_seed1.csv|omitted_raw_runs/final_experiments/paper_main_v1/gsm8k/TrustRoute_t1_0.75_t2_0.5_wq_0.4_wr_0.3_wc_0.3_mr_0.0_seed2.csv|omitted_raw_runs/final_experiments/paper_main_v1/gsm8k/TrustRoute_t1_0.75_t2_0.5_wq_0.4_wr_0.3_wc_0.3_mr_0.0_seed3.csv` |

## Evidence Matrix

| Reviewer question | Repository truth | Evidence |
|---|---|---|
| Does the main line use a real runtime judge? | No. The launcher passes `judge=None`; the dispatcher creates a constant stub; the TrustRoute core never invokes `judge_fn`. | [`run_exp.py#L277`](artifact/scripts/run_exp.py#L277), [`executor.py#L93`](src/runner/executor.py#L93), [`ours_lite_v2.py#L187`](src/runner/ours_lite_v2.py#L187) |
| Is routing based on latest observed cost/latency or EMA/rho? | Neither description matches the active code. Ranking uses static pricing metadata plus `quality` and reputation. | [`lite_utils.py#L134`](src/runner/lite_utils.py#L134), [`lite_utils.py#L144`](src/runner/lite_utils.py#L144) |
| What drives early stopping on non-code tasks? | Heuristic confidence computed from answer format. For MCQ tasks, a parsable option letter yields `0.90` confidence. | [`ours_lite_v2.py#L286`](src/runner/ours_lite_v2.py#L286), [`ours_lite_v2.py#L463`](src/runner/ours_lite_v2.py#L463) |
| Is GSM8K treated as executable verification? | No. GSM8K only gets a format hint; `tests_ok` is forced to remain false. | [`ours_lite_v2.py#L174`](src/runner/ours_lite_v2.py#L174), [`ours_lite_v2.py#L279`](src/runner/ours_lite_v2.py#L279) |
| How are stage-2 answers selected? | First by passed executable tests, then by majority vote, then by highest heuristic confidence. | [`ours_lite_v2.py#L395`](src/runner/ours_lite_v2.py#L395), [`ours_lite_v2.py#L401`](src/runner/ours_lite_v2.py#L401), [`ours_lite_v2.py#L411`](src/runner/ours_lite_v2.py#L411) |
| How is reputation persisted and isolated? | By `ISSTA_SEED` and `ISSTA_REP_TAG`, mapped to `ISSTA/cache/rep_state_<tag>_seed<seed>.json`. | [`run_exp.py#L105`](artifact/scripts/run_exp.py#L105), [`run_exp.py#L131`](artifact/scripts/run_exp.py#L131), [`lite_utils.py#L7`](src/runner/lite_utils.py#L7) |
| What updates reputation in stage 1 and stage 2? | Stage 1 uses test pass + confidence. Stage 2 uses test score + agreement + confidence. | [`ours_lite_v2.py#L587`](src/runner/ours_lite_v2.py#L587), [`ours_lite_v2.py#L619`](src/runner/ours_lite_v2.py#L619) |

## Dataset-Specific Runtime Truth

- Code tasks: `HumanEval` and `MBPP` receive code extraction and light execution tests. Only a true `pass` sets `tests_ok=True`.
- GSM8K: the code path performs a number/format hint only; it never marks the answer as verified executable evidence.
- MCQ tasks are explicitly listed as `['arc_challenge', 'hellaswag', 'mmlu', 'winogrande']` in [`ours_lite_v2.py#L11`](src/runner/ours_lite_v2.py#L11).
- For MCQ tasks, confidence is primarily driven by extracting a valid option letter, which is far weaker than an oracle-style verifier.

## Protocol Truth Extracted from `reviewer_truth_manifest.json`

| Fact | Value |
|---|---|
| Submission line | `7 benchmark + test90 + mixed-source TrustRoute` |
| `run_exp` passes `router=None` | `True` |
| `run_exp` passes `judge=None` | `True` |
| Dispatcher has judge stub | `True` |
| Judge stub returns constant | `True` |
| Ranking uses static pricing | `True` |
| Ranking uses rep state | `True` |
| Observed latency EMA exists | `False` |

## Reviewer-Safe Implications

- The current repository supports a strong claim about bounded escalation with online reputation and executable validation on code tasks.
- The current repository does not support a claim that an active judge model drives the main result line.
- The current repository does not support the paper's conflicting `latest observed` versus `EMA/rho` wording.
- The current repository does not support framing MCQ gains as execution-oracle gains.
- The current repository does support transparent disclosure of `test90`, per-dataset source selection, and per-dataset parameter provenance.

