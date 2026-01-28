#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT="${EXP_ROOT:-/data_huawei/jiakun/ISSTA/11111A_rerun_adj_para}"

SEEDS="${SEEDS:-1 2 3}"
DATASETS="${DATASETS:-hellaswag winogrande mmlu gsm8k}"
METHODS="${METHODS:-TrustRoute-NoRep TrustRoute-NoCost TrustRoute-NoCostAware TrustRoute-NoParallel}"

PARALLEL="${PARALLEL:-6}"

export EXP_ROOT
export SEEDS
export DATASETS
export METHODS

BUDGET_USD="${BUDGET_USD:-5.0}"
ETA="${ETA:-0.3}"
DEFAULT_MAX_K="${DEFAULT_MAX_K:-3}"

RUN_EXP_PY="$EXP_ROOT/scripts/run_exp.py"
BEST_CFG_DIR="$EXP_ROOT/best_cfg"
TEST90_DIR="$EXP_ROOT/test90"
OUT_ROOT="$EXP_ROOT/test90_runs"
LOG_ROOT="$EXP_ROOT/logs/test90"
SUMMARY_DIR="$EXP_ROOT/test90_summary"

if [ ! -f "$RUN_EXP_PY" ]; then
  echo "Missing: $RUN_EXP_PY" >&2
  exit 2
fi

if [ ! -d "$BEST_CFG_DIR" ]; then
  echo "Missing dir: $BEST_CFG_DIR" >&2
  exit 2
fi

if [ ! -d "$TEST90_DIR" ]; then
  echo "Missing dir: $TEST90_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$SUMMARY_DIR"

PIDS=()

_cleanup_bg() {
  if [ "${#PIDS[@]}" -gt 0 ]; then
    for pid in "${PIDS[@]}"; do
      kill "$pid" >/dev/null 2>&1 || true
    done
  fi
}

trap _cleanup_bg INT TERM
trap 'rc=$?; _cleanup_bg; exit $rc' ERR

_wait_one() {
  if [ "${#PIDS[@]}" -eq 0 ]; then
    return 0
  fi
  local pid="${PIDS[0]}"
  if ! wait "$pid"; then
    echo "A job failed (pid=$pid). Stopping." >&2
    exit 1
  fi
  PIDS=("${PIDS[@]:1}")
}

_wait_for_slot() {
  while [ "${#PIDS[@]}" -ge "$PARALLEL" ]; do
    _wait_one
  done
}

_extract_best_cfg() {
  local ds="$1"
  local cfg="$BEST_CFG_DIR/${ds}_best.json"
  if [ ! -f "$cfg" ]; then
    echo "Missing best_cfg: $cfg" >&2
    return 2
  fi

  python - <<PY
import json
p = r"$cfg"
with open(p, 'r', encoding='utf-8') as f:
    d = json.load(f)
print(
    str(d['tau1']),
    str(d['tau2']),
    str(d['w_q']),
    str(d['w_r']),
    str(d['w_c']),
    str(d.get('min_rep', 0.0)),
)
PY
}

for ds in $DATASETS; do
  input="$TEST90_DIR/${ds}_test90.jsonl"
  if [ ! -f "$input" ]; then
    echo "Missing input: $input" >&2
    exit 2
  fi

  read -r TAU1 TAU2 WQ WR WC MR < <(_extract_best_cfg "$ds")

  mkdir -p "$OUT_ROOT/$ds" "$LOG_ROOT/$ds"

  for seed in $SEEDS; do
    for method in $METHODS; do
      max_k="$DEFAULT_MAX_K"
      if [ "$method" = "TrustRoute-NoParallel" ]; then
        max_k=1
      fi

      out_csv="$OUT_ROOT/$ds/${ds}_${method}_bestcfg_seed${seed}.csv"
      log_fp="$LOG_ROOT/$ds/${ds}_${method}_bestcfg_seed${seed}.log"

      _wait_for_slot

      (
        echo "[RUN] ds=$ds seed=$seed method=$method max_k=$max_k" | tee -a "$log_fp"

        ISSTA_REP_TAG="${ds}_${method}_bestcfg_seed${seed}" \
        python -u "$RUN_EXP_PY" \
          --method "$method" \
          --input "$input" \
          --seed "$seed" \
          --tau1 "$TAU1" --tau2 "$TAU2" \
          --w_q "$WQ" --w_r "$WR" --w_c "$WC" --min_rep "$MR" \
          --max_k "$max_k" --budget_usd "$BUDGET_USD" --eta "$ETA" \
          --out "$out_csv" \
          --log_file "$log_fp"
      ) &
      PIDS+=("$!")
    done
  done
done

while [ "${#PIDS[@]}" -gt 0 ]; do
  _wait_one
done

OUT_SUMMARY="$SUMMARY_DIR/ablation_bestcfg_summary.csv"

python - <<'PY'
import csv
import statistics
from pathlib import Path
import importlib.util

EXP_ROOT = Path('/data_huawei/jiakun/ISSTA/11111A_rerun_adj_para')
EXP_ROOT = Path(__import__('os').environ.get('EXP_ROOT', str(EXP_ROOT)))

TEST90_DIR = EXP_ROOT / 'test90'
OUT_ROOT = EXP_ROOT / 'test90_runs'
OUT_SUMMARY = EXP_ROOT / 'test90_summary' / 'ablation_bestcfg_summary.csv'

DATASETS = (__import__('os').environ.get('DATASETS') or 'hellaswag winogrande mmlu gsm8k').split()
METHODS = (__import__('os').environ.get('METHODS') or 'TrustRoute-NoRep TrustRoute-NoCost TrustRoute-NoCostAware TrustRoute-NoParallel').split()
SEEDS = [int(x) for x in (__import__('os').environ.get('SEEDS') or '1 2 3').split()]

spec = importlib.util.spec_from_file_location('summ', str(EXP_ROOT / 'scripts' / 'summarize_test90.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

TIMEOUT = {'humaneval': 10, 'mbpp': 10}

def mean(xs):
    return statistics.fmean(xs) if xs else 0.0

def pstdev(xs):
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0

rows_out = []

for ds in DATASETS:
    gt_path = TEST90_DIR / f'{ds}_test90.jsonl'
    if not gt_path.exists():
        raise SystemExit(f'Missing test90 gt: {gt_path}')

    gt_order, gt_map = mod._load_gt(ds, gt_path)

    for method in METHODS:
        per_seed = []
        for seed in SEEDS:
            csv_path = OUT_ROOT / ds / f'{ds}_{method}_bestcfg_seed{seed}.csv'
            if not csv_path.exists():
                continue
            r = mod.summarize_one_csv(
                ds,
                csv_path,
                gt_order,
                gt_map,
                timeout_s=TIMEOUT.get(ds, 0),
                label=f'{ds}:{method}:seed{seed}',
            )
            per_seed.append(r)

        if not per_seed:
            continue

        accs = [r['acc'] for r in per_seed]
        costs = [r['cost'] for r in per_seed]
        lats = [r['lat'] for r in per_seed]
        nps = [r.get('n_present', 0) for r in per_seed]

        rows_out.append({
            'dataset': ds,
            'method': method,
            'n_seeds': len(per_seed),
            'acc_mean': mean(accs),
            'acc_std': pstdev(accs),
            'cost_mean': mean(costs),
            'cost_std': pstdev(costs),
            'lat_mean': mean(lats),
            'lat_std': pstdev(lats),
            'n_present_mean': mean(nps),
            'n_total': len(gt_order),
        })

OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

with OUT_SUMMARY.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=[
        'dataset','method','n_seeds',
        'acc_mean','acc_std',
        'cost_mean','cost_std',
        'lat_mean','lat_std',
        'n_present_mean','n_total'
    ])
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print('\n===== Ablation Summary (best_cfg, test90) =====')
print(f'Wrote: {OUT_SUMMARY}')

for r in sorted(rows_out, key=lambda x: (x['dataset'], x['method'])):
    print(
        f"{r['dataset']:12s} | {r['method']:20s} | seeds={r['n_seeds']} | "
        f"acc={r['acc_mean']:.4f}±{r['acc_std']:.4f} | "
        f"cost={r['cost_mean']:.6f}±{r['cost_std']:.6f} | "
        f"lat={r['lat_mean']:.2f}±{r['lat_std']:.2f}"
    )
print('=============================================\n')
PY

echo "ALL DONE. Summary: $OUT_SUMMARY"
