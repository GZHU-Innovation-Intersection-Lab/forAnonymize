import argparse
import csv
import os
import re
from pathlib import Path


def _default_exp_root() -> Path:
    env = os.environ.get("EXP_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def _read_jsonl_task_ids(path: Path):
    order = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = __import__("json").loads(line)
            except Exception:
                continue
            tid = obj.get("task_id") or obj.get("id") or f"{path.stem}-{i}"
            tid = str(tid)
            if tid not in seen:
                order.append(tid)
                seen.add(tid)
    return order


def _read_last_rows_by_task_id(csv_path: Path):
    last = {}
    fieldnames = None
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return None, None
        for row in reader:
            tid = str(row.get("task_id") or "").strip()
            if tid:
                last[tid] = row
    return fieldnames, last


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp_root", default="")
    ap.add_argument(
        "--selected_dir",
        default="/data_huawei/jiakun/ISSTA/final_experiments/selected",
    )
    ap.add_argument(
        "--evaluated_dir",
        default="/data_huawei/jiakun/ISSTA/final_experiments/evaluated",
    )
    ap.add_argument("--prefer_evaluated", action="store_true")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--methods", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    exp_root = Path(args.exp_root).resolve() if args.exp_root else _default_exp_root()

    test_jsonl = exp_root / "test90" / f"{args.dataset}_test90.jsonl"
    if not test_jsonl.exists():
        raise SystemExit(f"Missing test90 jsonl: {test_jsonl}")

    test_order = _read_jsonl_task_ids(test_jsonl)
    test_set = set(test_order)

    selected_dir = Path(args.selected_dir).resolve()
    if not selected_dir.exists():
        raise SystemExit(f"Missing selected_dir: {selected_dir}")

    evaluated_dir = Path(args.evaluated_dir).resolve()

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else (exp_root / "baseline_test90" / args.dataset)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    allow_methods = set([m.strip() for m in args.methods if str(m).strip()])

    pat = re.compile(rf"^{re.escape(args.dataset)}_(.+)_seed(\d+)\.csv$")

    total_written = 0
    for seed in args.seeds:
        seed = int(seed)
        matched = sorted(selected_dir.glob(f"{args.dataset}_*_seed{seed}.csv"))
        matched = [p for p in matched if p.is_file() and p.parent == selected_dir]

        for src in matched:
            m = pat.match(src.name)
            if not m:
                continue
            method = m.group(1)
            if allow_methods and method not in allow_methods:
                continue

            src_use = src
            if args.prefer_evaluated and evaluated_dir.exists():
                cand = evaluated_dir / src.name
                if cand.exists() and cand.is_file():
                    src_use = cand

            dst = out_dir / src.name
            if (not args.force) and dst.exists() and dst.stat().st_size > 100:
                continue

            fieldnames, last_map = _read_last_rows_by_task_id(src_use)
            if not fieldnames or last_map is None:
                continue

            with dst.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                wrote = 0
                for tid in test_order:
                    row = last_map.get(tid)
                    if not row:
                        continue
                    w.writerow(row)
                    wrote += 1

            total_written += 1
            print(
                f"[OK] {args.dataset} seed{seed} {method}: wrote {dst} ({wrote}/{len(test_order)})"
            )

    print(f"[DONE] {args.dataset}: outputs={total_written} out_dir={out_dir}")


if __name__ == "__main__":
    main()
