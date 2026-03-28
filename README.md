# TrustRoute GitHub Release Project

This directory is a cleaned GitHub-ready subset of the TrustRoute workspace.
It contains only our project assets:

- TrustRoute source code
- configuration files
- curated benchmark inputs/splits used by our pipeline
- official TrustRoute result summaries
- our ablations and TrustRoute-specific control variants
- paper source snapshot

It intentionally excludes third-party comparison experiment packages and their result trees.

## Included

- `src/`: core TrustRoute implementation
- `configs/`: agent/configuration files
- `artifact/inputs`, `artifact/splits`, `artifact/dev10`, `artifact/test90`: benchmark inputs and splits
- `artifact/best_cfg`: selected TrustRoute configuration files
- `artifact/scripts`: curated scripts for running TrustRoute and rebuilding our summaries
- `results/official`: official TrustRoute summaries and paper-safe trade-off files
- `results/official`: TrustRoute mainline and TrustRoute+ablation summaries only
- `results/audits`: protocol, runtime, and fairness audits
- `results/ablations`: TrustRoute ablations and TrustRoute-only control variants
- `paper/`: current paper PDF and LaTeX snapshot

## Explicitly Excluded

The following are intentionally not copied into this GitHub release directory:

- baseline result trees for other methods
- recent-router comparison result trees:
  - `CP-Router`
  - `Smoothie`
  - `AutoMixStyle`
- execution-aware comparison baselines that are not part of the core TrustRoute package:
  - `CodeExecRerank`
  - `CodeSelfRepair`
- response bundles, attachment build intermediates, screenshots, and other rebuttal-only packaging files
- raw run logs, caches, worker outputs, and backup directories

## Key Entry Points

- Main TrustRoute runtime: [src/runner/ours_lite_v2.py](src/runner/ours_lite_v2.py)
- Main executor: [src/runner/executor.py](src/runner/executor.py)
- Agent pool loader: [src/agents/pool.py](src/agents/pool.py)
- Agent config: [configs/agents.yaml](configs/agents.yaml)
- Experiment runner: [artifact/scripts/run_exp.py](artifact/scripts/run_exp.py)

## Result Entry Points

- Official mainline summary: [results/official/trustroute_mainline_only.csv](results/official/trustroute_mainline_only.csv)
- TrustRoute + ablations summary: [results/official/trustroute_full_and_ablations_test90.csv](results/official/trustroute_full_and_ablations_test90.csv)
- Protocol manifest: [results/official/protocol_manifest_test90.csv](results/official/protocol_manifest_test90.csv)
- Truth manifest: [results/official/reviewer_truth_manifest.json](results/official/reviewer_truth_manifest.json)
- Judge usage audit: [results/audits/judge_usage_audit.md](results/audits/judge_usage_audit.md)
- Routing diagnostics: [results/audits/routing_diagnostics.md](results/audits/routing_diagnostics.md)
- Execution ablation: [results/ablations/TrustRoute-NoLightTests](results/ablations/TrustRoute-NoLightTests)

## Notes

- This directory is a staging tree for GitHub publication; it does not modify the original workspace.
- Comparison-only summary files have been removed from this tree; only TrustRoute-owned results and ablations remain.
- Raw worker logs, third-party comparison outputs, and omitted private run artifacts are intentionally excluded from this tree.
- API-backed re-execution still requires valid credentials in the environment expected by `configs/agents.yaml`.
