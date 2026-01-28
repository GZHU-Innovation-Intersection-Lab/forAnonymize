#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT_DEFAULT="/data_huawei/jiakun/ISSTA/11111A_rerun_adj_para"
EXP_ROOT="${EXP_ROOT:-$EXP_ROOT_DEFAULT}"

SEEDS="1"
LIMIT=0
TIMEOUT_SEC=9000
CLEAN_REP_STATE=0
DATASETS="arc_challenge hellaswag winogrande mmlu humaneval mbpp gsm8k"
SELECTED_DIR="/data_huawei/jiakun/ISSTA/final_experiments/selected"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_root)
      EXP_ROOT="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --timeout_sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    --clean_rep_state)
      CLEAN_REP_STATE=1
      shift 1
      ;;
    --datasets)
      DATASETS="$2"
      shift 2
      ;;
    --selected_dir)
      SELECTED_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_BEST_PY="$SCRIPT_DIR/run_test90_best.py"
FILTER_BASE_PY="$SCRIPT_DIR/filter_baselines_test90.py"
SUMMARIZE_PY="$SCRIPT_DIR/summarize_test90.py"

if [[ ! -f "$RUN_BEST_PY" ]]; then
  echo "Missing $RUN_BEST_PY" >&2
  exit 1
fi
if [[ ! -f "$FILTER_BASE_PY" ]]; then
  echo "Missing $FILTER_BASE_PY" >&2
  exit 1
fi
if [[ ! -f "$SUMMARIZE_PY" ]]; then
  echo "Missing $SUMMARIZE_PY" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG_DIR="$EXP_ROOT/logs/test90/_driver"
mkdir -p "$DRIVER_LOG_DIR" "$EXP_ROOT/logs/test90" "$EXP_ROOT/test90_runs" "$EXP_ROOT/baseline_test90" "$EXP_ROOT/test90_summary"
MASTER_LOG="$DRIVER_LOG_DIR/run_all_test90_${TS}.log"

{
  echo "EXP_ROOT=$EXP_ROOT"
  echo "SEEDS=$SEEDS"
  echo "LIMIT=$LIMIT"
  echo "TIMEOUT_SEC=$TIMEOUT_SEC"
  echo "CLEAN_REP_STATE=$CLEAN_REP_STATE"
  echo "DATASETS=$DATASETS"
  echo "SELECTED_DIR=$SELECTED_DIR"
  echo "START_TS=$TS"
  echo
} | tee -a "$MASTER_LOG"

for ds in $DATASETS; do
  test_input="$EXP_ROOT/test90/${ds}_test90.jsonl"
  best_cfg="$EXP_ROOT/best_cfg/${ds}_best.json"
  ds_log="$DRIVER_LOG_DIR/${ds}_${TS}.log"

  if [[ ! -f "$test_input" ]]; then
    echo "[FATAL] Missing test90 input: $test_input" | tee -a "$MASTER_LOG" "$ds_log" >&2
    exit 1
  fi
  if [[ ! -f "$best_cfg" ]]; then
    echo "[FATAL] Missing best_cfg: $best_cfg" | tee -a "$MASTER_LOG" "$ds_log" >&2
    exit 1
  fi

  mkdir -p "$EXP_ROOT/test90_runs/$ds" "$EXP_ROOT/logs/test90/$ds" "$EXP_ROOT/baseline_test90/$ds" "$EXP_ROOT/test90_summary"

  {
    echo "="
    echo "[DATASET] $ds"
    echo "[TIME] $(date +%F' '%T)"
  } | tee -a "$MASTER_LOG" "$ds_log"

  for seed in $SEEDS; do
    cmd=(
      python -u "$RUN_BEST_PY"
      --dataset "$ds"
      --exp_root "$EXP_ROOT"
      --seed "$seed"
      --timeout_sec "$TIMEOUT_SEC"
    )

    if [[ "$CLEAN_REP_STATE" == "1" ]]; then
      cmd+=(--clean_rep_state)
    fi
    if [[ "$LIMIT" != "0" ]]; then
      cmd+=(--limit "$LIMIT")
    fi

    echo "[RUN_BEST] ${cmd[*]}" | tee -a "$MASTER_LOG" "$ds_log"
    "${cmd[@]}" 2>&1 | tee -a "$MASTER_LOG" "$ds_log"
  done

  base_cmd=(
    python -u "$FILTER_BASE_PY"
    --dataset "$ds"
    --exp_root "$EXP_ROOT"
    --selected_dir "$SELECTED_DIR"
    --prefer_evaluated
    --seeds $SEEDS
  )
  echo "[FILTER_BASELINES] ${base_cmd[*]}" | tee -a "$MASTER_LOG" "$ds_log"
  "${base_cmd[@]}" 2>&1 | tee -a "$MASTER_LOG" "$ds_log"

  sum_cmd=(
    python -u "$SUMMARIZE_PY"
    --dataset "$ds"
    --exp_root "$EXP_ROOT"
    --timeout_s 10
    --include_per_seed
  )
  echo "[SUMMARIZE] ${sum_cmd[*]}" | tee -a "$MASTER_LOG" "$ds_log"
  "${sum_cmd[@]}" 2>&1 | tee -a "$MASTER_LOG" "$ds_log"

done

echo "ALL DONE. MASTER_LOG=$MASTER_LOG" | tee -a "$MASTER_LOG"
