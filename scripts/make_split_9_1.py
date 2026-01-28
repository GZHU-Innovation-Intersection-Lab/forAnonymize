import argparse
import hashlib
import json
import math
from pathlib import Path


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--skip_if_exists", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    dev_dir = out_root / "dev10"
    test_dir = out_root / "test90"
    split_dir = out_root / "splits"
    dev_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    out_dev = dev_dir / f"{args.dataset}_dev10.jsonl"
    out_test = test_dir / f"{args.dataset}_test90.jsonl"
    out_dev_ids = split_dir / f"{args.dataset}_dev_ids.txt"
    out_test_ids = split_dir / f"{args.dataset}_test_ids.txt"

    if (
        args.skip_if_exists
        and out_dev.exists()
        and out_test.exists()
        and out_dev_ids.exists()
        and out_test_ids.exists()
    ):
        print(f"[SKIP] {args.dataset}: outputs already exist.")
        return

    rows = []
    inp = Path(args.input)
    with inp.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            tid = obj.get("task_id") or obj.get("id") or f"{args.dataset}-{idx}"
            tid = str(tid)
            obj["task_id"] = tid

            h = stable_hash(f"{tid}::seed={args.seed}")
            rows.append((h, tid, obj))

    n = len(rows)
    if n == 0:
        raise RuntimeError(f"No rows found in {inp}")

    rows.sort(key=lambda x: (x[0], x[1]))
    n_dev = max(1, int(math.floor(n * args.dev_frac)))

    dev_rows = [r[2] for r in rows[:n_dev]]
    test_rows = [r[2] for r in rows[n_dev:]]

    with out_dev.open("w", encoding="utf-8") as fdev, out_test.open(
        "w", encoding="utf-8"
    ) as ftest:
        for obj in dev_rows:
            fdev.write(json.dumps(obj, ensure_ascii=False) + "\n")
        for obj in test_rows:
            ftest.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with out_dev_ids.open("w", encoding="utf-8") as f:
        for obj in dev_rows:
            f.write(str(obj["task_id"]) + "\n")

    with out_test_ids.open("w", encoding="utf-8") as f:
        for obj in test_rows:
            f.write(str(obj["task_id"]) + "\n")

    print(f"[OK] {args.dataset}: total={n}, dev={len(dev_rows)}, test={len(test_rows)}")
    print(f"     dev:  {out_dev}")
    print(f"     test: {out_test}")
    print(f"     ids:  {out_dev_ids}, {out_test_ids}")


if __name__ == "__main__":
    main()
