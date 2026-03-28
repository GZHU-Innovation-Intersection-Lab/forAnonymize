# Non-Executable Runtime Profile

This file summarizes how the official TrustRoute line behaves on datasets where the current code does not have a strong executable oracle.

## Bottom Line

- These datasets are not powered by a hidden runtime judge in the active code path.
- They are mainly driven by stage-1 routing, heuristic confidence, bounded stage-2 escalation, and majority-vote style answer extraction.
- This profile is meant to answer reviewer skepticism with runtime evidence, not to overclaim stronger verification than the code actually performs.

## Official TrustRoute Runtime Profile

| Dataset | Mode | Stage1 | Stage2 Vote | Stage2 Conf | Stage2 Verified | Top Reason | Top Agent |
|---|---|---:|---:|---:|---:|---|---|
| arc_challenge | MCQ letter heuristic + majority vote | 98.9% | 0.8% | 0.3% | 0.0% | `stage1_early_stop` (98.9%) | `cand-gpt-5-mini` (99.5%) |
| gsm8k | format-only confidence + majority vote | 0.0% | 95.0% | 5.0% | 0.0% | `stage2_vote` (95.0%) | `cand-gpt-4o-mini` (91.9%) |
| hellaswag | MCQ letter heuristic + majority vote | 38.6% | 39.1% | 22.4% | 0.0% | `stage2_vote` (39.1%) | `cand-gemini-2.5-flash` (41.8%) |
| mmlu | MCQ letter heuristic + majority vote | 99.5% | 0.3% | 0.3% | 0.0% | `stage1_early_stop` (99.5%) | `cand-gemini-2.5-flash` (66.6%) |
| winogrande | MCQ letter heuristic + majority vote | 97.1% | 1.1% | 1.7% | 0.0% | `stage1_early_stop` (97.1%) | `cand-gemini-2.5-flash` (66.5%) |

## Interpretation

- If `stage1_early_stop` dominates, the gain is primarily coming from route-to-strong-single-model behavior.
- If `stage2_vote` or `stage2_confidence` are substantial, the gain is coming from bounded multi-agent escalation rather than a hidden judge.
- `stage2_verified` is expected to stay absent on MCQ datasets because the current code does not have an executable verifier there.

## Evidence

- Machine-readable summary: [`non_executable_runtime_profile.csv`](results/audits/non_executable_runtime_profile.csv)
- Main source selection: [`protocol_manifest_test90.csv`](results/official/protocol_manifest_test90.csv)
- Runtime method audit: [`runtime_method_audit.md`](results/audits/runtime_method_audit.md)

