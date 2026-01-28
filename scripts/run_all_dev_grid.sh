#!/usr/bin/env bash
set -euo pipefail

EXP_ROOT_DEFAULT="/data_huawei/jiakun/ISSTA/11111A_rerun_adj_para"
EXP_ROOT="${EXP_ROOT:-$EXP_ROOT_DEFAULT}"

SEED=1
LIMIT=0
TIMEOUT_SEC=9000
MAX_WORKERS=6
CLEAN_REP_STATE=0
DATASETS="arc_challenge hellaswag winogrande mmlu humaneval mbpp gsm8k"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp_root)
      EXP_ROOT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
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
    --max_workers)
      MAX_WORKERS="$2"
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
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DEV_GRID_PY="$SCRIPT_DIR/run_dev_grid.py"
SUMMARIZE_PY="$SCRIPT_DIR/summarize_dev_grid.py"

if [[ ! -f "$RUN_DEV_GRID_PY" ]]; then
  echo "Missing $RUN_DEV_GRID_PY" >&2
  exit 1
fi

if [[ ! -f "$SUMMARIZE_PY" ]]; then
  echo "Missing $SUMMARIZE_PY" >&2
  exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG_DIR="$EXP_ROOT/logs/dev_grid/_driver"
mkdir -p \
  "$DRIVER_LOG_DIR" \
  "$EXP_ROOT/dev_grid" \
  "$EXP_ROOT/logs/dev_grid" \
  "$EXP_ROOT/logs/test90" \
  "$EXP_ROOT/dev_summary" \
  "$EXP_ROOT/best_cfg" \
  "$EXP_ROOT/splits"

MASTER_LOG="$DRIVER_LOG_DIR/run_all_dev_grid_${TS}.log"

{
  echo "EXP_ROOT=$EXP_ROOT"
  echo "SEED=$SEED"
  echo "LIMIT=$LIMIT"
  echo "TIMEOUT_SEC=$TIMEOUT_SEC"
  echo "MAX_WORKERS=$MAX_WORKERS"
  echo "CLEAN_REP_STATE=$CLEAN_REP_STATE"
  echo "DATASETS=$DATASETS"
  echo "START_TS=$TS"
  echo
} | tee -a "$MASTER_LOG"

for ds in $DATASETS; do
  input="$EXP_ROOT/dev10/${ds}_dev10.jsonl"
  out_dir="$EXP_ROOT/dev_grid/${ds}"
  log_dir="$EXP_ROOT/logs/dev_grid/${ds}"
  baseline_dir="$EXP_ROOT/baseline_test90/${ds}"
  test90_log_dir="$EXP_ROOT/logs/test90/${ds}"
  ds_log="$DRIVER_LOG_DIR/${ds}_${TS}.log"

  if [[ ! -f "$input" ]]; then
    echo "[FATAL] Missing input: $input" | tee -a "$MASTER_LOG" "$ds_log" >&2
    exit 1
  fi

  mkdir -p "$out_dir" "$log_dir" "$baseline_dir" "$test90_log_dir"

  cmd=(
    python -u "$RUN_DEV_GRID_PY"
    --dataset "$ds"
    --input "$input"
    --seed "$SEED"
    --rep_state_scope per_run
    --max_workers "$MAX_WORKERS"
    --timeout_sec "$TIMEOUT_SEC"
    --out_dir "$out_dir"
    --log_dir "$log_dir"
  )

  if [[ "$CLEAN_REP_STATE" == "1" ]]; then
    cmd+=(--clean_rep_state)
  fi

  if [[ "$LIMIT" != "0" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  {
    echo "="
    echo "[DATASET] $ds"
    echo "[TIME] $(date +%F' '%T)"
    echo "[CMD] ${cmd[*]}"
  } | tee -a "$MASTER_LOG" "$ds_log"

  "${cmd[@]}" 2>&1 | tee -a "$MASTER_LOG" "$ds_log"

  csv_count=0
  if ls "$out_dir"/*.csv >/dev/null 2>&1; then
    csv_count="$(ls -1 "$out_dir"/*.csv | wc -l | tr -d ' ')"
  fi

  {
    echo "[DONE] $ds"
    echo "[CSV_DIR] $out_dir"
    echo "[CSV_COUNT] $csv_count"
    echo "[LOG_DIR] $log_dir"
    echo "[SUMMARY] $EXP_ROOT/dev_summary/${ds}_grid_summary.csv"
    echo "[BEST_CFG] $EXP_ROOT/best_cfg/${ds}_best.json"
    echo "[DRIVER_LOG] $ds_log"
    echo
  } | tee -a "$MASTER_LOG" "$ds_log"

  python -u "$SUMMARIZE_PY" --dataset "$ds" --exp_root "$EXP_ROOT" --timeout_s 10 2>&1 | tee -a "$MASTER_LOG" "$ds_log"

done

echo "ALL DONE. MASTER_LOG=$MASTER_LOG" | tee -a "$MASTER_LOG"
