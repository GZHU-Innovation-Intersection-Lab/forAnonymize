import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


MCQ_DATASETS = {"hellaswag", "arc_challenge", "winogrande", "mmlu"}


def _read_jsonl(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _extract_last_number(text: str):
    if text is None:
        return None
    s = str(text)
    s = s.replace(",", "")

    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", s)
    if m:
        return str(m[-1]).strip()

    m = re.findall(r"\\boxed\{\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)\s*\}", s)
    if m:
        return str(m[-1]).strip()

    m = re.findall(
        r"(?is)(?:final\s+answer|answer|ans|答案|最终答案)\s*(?:is|=|:|：|是)?\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)",
        s,
    )
    if m:
        return str(m[-1]).strip()

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None


def _parse_number(x: str):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.match(r"^\s*([-+]?\d+)\s*/\s*(\d+)\s*$", s)
    if m:
        try:
            a = float(m.group(1))
            b = float(m.group(2))
            if abs(b) < 1e-12:
                return None
            return a / b
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _extract_mcq_letter(text: str, dataset: str):
    allow = "AB" if dataset == "winogrande" else "ABCD"
    t = ("" if text is None else str(text)).strip().upper()
    m = re.search(r"ANSWER\s*[:：]\s*([%s])" % allow, t)
    if m:
        return m.group(1)
    m2 = re.findall(r"(?:^|\b)([%s])(?:\b|$)" % allow, t)
    return m2[-1] if m2 else None


def _extract_code(text: str) -> str:
    if text is None:
        return ""
    s = str(text)
    m = re.findall(r"```(?:python)?\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return max(m, key=len).strip()
    return s.strip()


def _humaneval_test_code(test: str, entry_point: str) -> str:
    t = test or ""
    if re.search(r"(?m)^\s*(?!def\s)(check|test_check)\s*\(", t):
        return t.strip() + "\n"
    if re.search(r"(?m)^\s*def\s+test_check\s*\(", t):
        t = t.rstrip() + "\n\ntry:\n    test_check()\nexcept NameError:\n    pass\n"
        return t.strip() + "\n"
    if re.search(r"(?m)^\s*def\s+check\s*\(", t) and entry_point:
        t = t.rstrip() + f"\n\ncheck({entry_point})\n"
        return t.strip() + "\n"
    return t.strip() + "\n"


def _mbpp_test_code(obj: dict) -> str:
    parts = []
    c = obj.get("contract")
    if isinstance(c, str) and c.strip():
        for line in c.splitlines():
            if re.match(r"^\s*(import\s+\w|from\s+\w)", line):
                parts.append(line)

    a = obj.get("assertion")
    if isinstance(a, list):
        parts.extend([str(x) for x in a if str(x).strip()])
    elif isinstance(a, str) and a.strip():
        parts.append(a)

    t = obj.get("test_list")
    if isinstance(t, list):
        parts.extend([str(x) for x in t if str(x).strip()])
    elif isinstance(t, str) and t.strip():
        parts.append(t)

    test = "\n".join(parts).replace("$_CONTRACT_$", "")
    return test.strip() + "\n" if test.strip() else ""


def _sandbox_eval(user_code: str, test_code: str, timeout_s: int):
    if not user_code.strip() or not test_code.strip():
        return False

    wrapper = f"""
import os, tempfile, signal

def _timeout_handler(signum, frame):
    raise TimeoutError('timeout')

signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm({int(timeout_s) + 1})

user_code = {user_code!r}
test_code = {test_code!r}

with tempfile.TemporaryDirectory(prefix='test90_sandbox_') as td:
    os.chdir(td)
    g = {{'__name__': '__main__'}}
    exec(compile(user_code, '<user_code>', 'exec'), g, g)
    exec(compile(test_code, '<test_code>', 'exec'), g, g)

signal.alarm(0)
"""

    try:
        cp = subprocess.run(
            [sys.executable, "-S", "-c", wrapper],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=int(timeout_s) + 2,
            env={"PYTHONNOUSERSITE": "1"},
        )
        return cp.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _read_last_rows_by_task_id(csv_path: Path):
    last = {}
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get("task_id") or "").strip()
            if tid:
                last[tid] = row
    return last


def _to_float(x):
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if not s:
            return 0.0
        return float(s)
    except Exception:
        return 0.0


def _load_gt(dataset: str, test_jsonl: Path):
    items = _read_jsonl(test_jsonl)
    gt_order = []
    gt_map = {}
    for i, obj in enumerate(items):
        tid = obj.get("task_id") or obj.get("id") or f"{dataset}-{i}"
        tid = str(tid)
        gt_order.append(tid)
        gt_map[tid] = obj
    return gt_order, gt_map


def _infer_seed_from_name(name: str):
    m = re.search(r"_seed(\d+)\.csv$", Path(name).name)
    if m:
        return int(m.group(1))
    return None


def summarize_one_csv(dataset: str, csv_path: Path, gt_order, gt_map, timeout_s: int, label: str):
    last_rows = _read_last_rows_by_task_id(csv_path)

    n_total = len(gt_order)
    n_present = 0
    n_correct = 0
    costs = []
    lats = []

    for tid in gt_order:
        row = last_rows.get(tid)
        if not row:
            continue

        err = str(row.get("error") or "").strip()
        ans = row.get("answer")
        ans_s = "" if ans is None else str(ans).strip()
        if err or (not ans_s):
            continue

        n_present += 1
        costs.append(_to_float(row.get("cost_usd")))
        lats.append(_to_float(row.get("latency_s")))

        gt_obj = gt_map.get(tid, {})

        ok = False
        if "is_correct" in row:
            v = str(row.get("is_correct") or "").strip().lower()
            if v in {"1", "true", "t", "yes", "y"}:
                ok = True
            elif v in {"0", "false", "f", "no", "n"}:
                ok = False
            else:
                ok = False

        if dataset == "gsm8k":
            pred = _parse_number(_extract_last_number(ans_s))
            exp = _parse_number(_extract_last_number(gt_obj.get("answer")))
            if pred is not None and exp is not None:
                ok = abs(float(pred) - float(exp)) < 1e-3

        elif dataset in MCQ_DATASETS:
            pred = _extract_mcq_letter(ans_s, dataset)
            exp = str(gt_obj.get("answer") or "").strip().upper()
            ok = (pred is not None) and (pred == exp)

        elif dataset == "humaneval":
            entry = str(gt_obj.get("entry_point") or "").strip()
            prompt = str(gt_obj.get("prompt") or "")
            test = str(gt_obj.get("test") or "")
            code = _extract_code(ans_s)
            if entry and re.search(rf"(?m)^\s*def\s+{re.escape(entry)}\s*\(", code):
                pre = ""
                if f"def {entry}" in prompt:
                    pre = prompt.split(f"def {entry}", 1)[0]
                user_code = (pre.rstrip() + "\n" + code).strip() + "\n"
            else:
                user_code = (prompt.rstrip() + "\n" + code).strip() + "\n"
            test_code = _humaneval_test_code(test, entry)
            if "is_correct" not in row:
                ok = _sandbox_eval(user_code, test_code, timeout_s)

        elif dataset == "mbpp":
            code = _extract_code(ans_s)
            test_code = _mbpp_test_code(gt_obj)
            if "is_correct" not in row:
                ok = _sandbox_eval(code + "\n", test_code, timeout_s)

        if ok:
            n_correct += 1

    acc = (n_correct / n_total) if n_total else 0.0
    cost_mean = statistics.fmean(costs) if costs else 0.0
    lat_mean = statistics.fmean(lats) if lats else 0.0

    seed = None
    if last_rows:
        any_row = next(iter(last_rows.values()))
        seed = any_row.get("seed")
        try:
            seed = int(seed) if seed is not None and str(seed).strip() else None
        except Exception:
            seed = None
    if seed is None:
        seed = _infer_seed_from_name(csv_path.name)

    return {
        "method": label,
        "seed": seed,
        "acc": acc,
        "cost": cost_mean,
        "lat": lat_mean,
        "n_present": float(n_present),
        "n_total": float(n_total),
        "file": str(csv_path),
    }


def _default_exp_root() -> Path:
    env = os.environ.get("EXP_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp_root", default="")
    ap.add_argument("--timeout_s", type=int, default=3)
    ap.add_argument("--baseline_dir", default="")
    ap.add_argument("--trustroute_best_dir", default="")
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--include_per_seed", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    exp_root = Path(args.exp_root).resolve() if args.exp_root else _default_exp_root()

    dataset = args.dataset
    test_jsonl = exp_root / "test90" / f"{dataset}_test90.jsonl"
    if not test_jsonl.exists():
        raise SystemExit(f"Missing test90 input: {test_jsonl}")

    baseline_dir = (
        Path(args.baseline_dir).resolve()
        if args.baseline_dir
        else (exp_root / "baseline_test90" / dataset)
    )
    trustroute_best_dir = (
        Path(args.trustroute_best_dir).resolve()
        if args.trustroute_best_dir
        else (exp_root / "test90_runs" / dataset)
    )

    gt_order, gt_map = _load_gt(dataset, test_jsonl)

    rows = []

    if trustroute_best_dir.exists():
        best_csvs = sorted([p for p in trustroute_best_dir.glob("*.csv") if p.is_file()])
        for p in best_csvs:
            rows.append(
                summarize_one_csv(
                    dataset,
                    p,
                    gt_order,
                    gt_map,
                    int(args.timeout_s),
                    label="TrustRoute(best_cfg)",
                )
            )

    if baseline_dir.exists():
        base_csvs = sorted([p for p in baseline_dir.glob("*.csv") if p.is_file()])
        for p in base_csvs:
            m = re.match(rf"^{re.escape(dataset)}_(.+)_seed(\d+)\.csv$", p.name)
            label = m.group(1) if m else "baseline"
            rows.append(
                summarize_one_csv(dataset, p, gt_order, gt_map, int(args.timeout_s), label=label)
            )

    if not rows:
        raise SystemExit("No CSVs found to summarize")

    grouped = {}
    for r in rows:
        grouped.setdefault(r["method"], []).append(r)

    out_csv = (
        Path(args.out_csv).resolve()
        if args.out_csv
        else (exp_root / "test90_summary" / f"{dataset}_test90_summary.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    def _mean(xs):
        return statistics.fmean(xs) if xs else 0.0

    def _std(xs):
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    summary_rows = []
    for method, items in grouped.items():
        items = sorted(items, key=lambda x: (x.get("seed") is None, x.get("seed") or 0))
        accs = [x["acc"] for x in items]
        costs = [x["cost"] for x in items]
        lats = [x["lat"] for x in items]
        pres = [x["n_present"] for x in items]
        ntot = items[0]["n_total"] if items else 0.0

        summary_rows.append(
            {
                "method": method,
                "n_seeds": len(items),
                "acc_mean": _mean(accs),
                "acc_std": _std(accs),
                "cost_mean": _mean(costs),
                "cost_std": _std(costs),
                "latency_mean": _mean(lats),
                "latency_std": _std(lats),
                "n_present_mean": _mean(pres),
                "n_total": ntot,
            }
        )

    summary_rows.sort(key=lambda x: (-x["acc_mean"], x["cost_mean"], x["latency_mean"]))

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "n_seeds",
                "acc_mean",
                "acc_std",
                "cost_mean",
                "cost_std",
                "latency_mean",
                "latency_std",
                "n_present_mean",
                "n_total",
            ]
        )
        for r in summary_rows:
            w.writerow(
                [
                    r["method"],
                    r["n_seeds"],
                    f"{r['acc_mean']:.12f}",
                    f"{r['acc_std']:.12f}",
                    f"{r['cost_mean']:.12f}",
                    f"{r['cost_std']:.12f}",
                    f"{r['latency_mean']:.12f}",
                    f"{r['latency_std']:.12f}",
                    f"{r['n_present_mean']:.1f}",
                    f"{r['n_total']:.1f}",
                ]
            )

    if args.include_per_seed:
        out_long = out_csv.with_name(out_csv.stem + "_per_seed.csv")
        with out_long.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "seed", "acc", "cost", "latency", "n_present", "n_total", "file"])
            for r in sorted(rows, key=lambda x: (x["method"], x.get("seed") or 0)):
                w.writerow(
                    [
                        r["method"],
                        r.get("seed"),
                        f"{r['acc']:.12f}",
                        f"{r['cost']:.12f}",
                        f"{r['lat']:.12f}",
                        f"{r['n_present']:.1f}",
                        f"{r['n_total']:.1f}",
                        r["file"],
                    ]
                )

    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
