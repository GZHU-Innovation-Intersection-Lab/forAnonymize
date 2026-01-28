import argparse
import json
import sys
from pathlib import Path


DEFAULT_DATASETS = [
    "gsm8k",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu",
    "humaneval",
    "mbpp",
]


def _count_lines(p: Path) -> int:
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def _exists_nonempty_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())


def _has_glob(p: Path, pat: str) -> bool:
    return any(p.glob(pat))


def _maybe_write_split_ids_from_jsonl(jsonl_path: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tid = obj.get("task_id") or obj.get("id") or f"row-{i}"
            fout.write(str(tid) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--write_missing_split_ids", action="store_true")
    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    datasets = [str(x) for x in args.datasets]

    missing = []

    def need(cond: bool, what: str):
        if not cond:
            missing.append(what)
            return False
        return True

    print(f"[CHECK] EXP_ROOT={exp_root}")

    base_dirs = [
        exp_root / "scripts",
        exp_root / "splits",
        exp_root / "dev10",
        exp_root / "test90",
        exp_root / "dev_grid",
        exp_root / "dev_summary",
        exp_root / "best_cfg",
        exp_root / "logs" / "dev_grid",
        exp_root / "logs" / "dev_grid" / "_driver",
        exp_root / "baseline_test90",
    ]

    for d in base_dirs:
        need(d.exists(), f"Missing dir: {d}")

    for ds in datasets:
        print(f"\n[DATASET] {ds}")

        dev_jsonl = exp_root / "dev10" / f"{ds}_dev10.jsonl"
        test_jsonl = exp_root / "test90" / f"{ds}_test90.jsonl"

        need(dev_jsonl.exists(), f"Missing dev jsonl: {dev_jsonl}")
        need(test_jsonl.exists(), f"Missing test jsonl: {test_jsonl}")

        dev_ids = exp_root / "splits" / f"{ds}_dev_ids.txt"
        test_ids = exp_root / "splits" / f"{ds}_test_ids.txt"

        if (not dev_ids.exists()) and args.write_missing_split_ids and dev_jsonl.exists():
            n = _maybe_write_split_ids_from_jsonl(dev_jsonl, dev_ids)
            print(f"[WRITE] {dev_ids} ({n} lines)")
        if (not test_ids.exists()) and args.write_missing_split_ids and test_jsonl.exists():
            n = _maybe_write_split_ids_from_jsonl(test_jsonl, test_ids)
            print(f"[WRITE] {test_ids} ({n} lines)")

        need(dev_ids.exists(), f"Missing split ids: {dev_ids}")
        need(test_ids.exists(), f"Missing split ids: {test_ids}")

        grid_dir = exp_root / "dev_grid" / ds
        log_dir = exp_root / "logs" / "dev_grid" / ds
        summary_csv = exp_root / "dev_summary" / f"{ds}_grid_summary.csv"
        best_json = exp_root / "best_cfg" / f"{ds}_best.json"

        if need(grid_dir.exists(), f"Missing dev_grid dir: {grid_dir}"):
            need(_has_glob(grid_dir, "*.csv"), f"No dev_grid csv under: {grid_dir}")

        if need(log_dir.exists(), f"Missing dev_grid logs dir: {log_dir}"):
            need(_has_glob(log_dir, "*.log"), f"No dev_grid logs under: {log_dir}")

        need(summary_csv.exists(), f"Missing dev_summary: {summary_csv}")
        need(best_json.exists(), f"Missing best_cfg: {best_json}")

        baseline_dir = exp_root / "baseline_test90" / ds
        need(baseline_dir.exists(), f"Missing baseline_test90 dir: {baseline_dir}")

        if dev_jsonl.exists():
            print(f"  dev_jsonl:  {dev_jsonl.name} ({_count_lines(dev_jsonl)})")
        if test_jsonl.exists():
            print(f"  test_jsonl: {test_jsonl.name} ({_count_lines(test_jsonl)})")
        if dev_ids.exists():
            print(f"  dev_ids:    {dev_ids.name} ({_count_lines(dev_ids)})")
        if test_ids.exists():
            print(f"  test_ids:   {test_ids.name} ({_count_lines(test_ids)})")

    if missing:
        print("\n[MISSING]")
        for m in missing:
            print(m)
        if args.strict:
            raise SystemExit(1)
    else:
        print("\n[OK] All checked artifacts exist.")


if __name__ == "__main__":
    main()
