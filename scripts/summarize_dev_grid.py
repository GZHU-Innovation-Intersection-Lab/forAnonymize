import argparse
import csv
import json
import math
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
    if "####" in s:
        s = s.split("####")[-1]
    s = s.replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None


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

with tempfile.TemporaryDirectory(prefix='dev_grid_sandbox_') as td:
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


def _parse_params_from_name(dataset: str, name: str):
    base = Path(name).name
    pat = rf"^{re.escape(dataset)}_TrustRoute_tau1_([\d\.]+)_tau2_([\d\.]+)_wq_([\d\.]+)_wr_([\d\.]+)_wc_([\d\.]+)_mr_([\d\.]+)(?:_seed(\d+))?\.csv$"
    m = re.match(pat, base)
    if not m:
        return None
    return {
        "t1": float(m.group(1)),
        "t2": float(m.group(2)),
        "wq": float(m.group(3)),
        "wr": float(m.group(4)),
        "wc": float(m.group(5)),
        "mr": float(m.group(6)),
        "seed": int(m.group(7)) if m.group(7) is not None else None,
    }


def _load_gt(dataset: str, dev_jsonl: Path):
    items = _read_jsonl(dev_jsonl)
    gt_order = []
    gt_map = {}
    for i, obj in enumerate(items):
        tid = obj.get("task_id") or obj.get("id") or f"{dataset}-{i}"
        tid = str(tid)
        gt_order.append(tid)
        gt_map[tid] = obj
    return gt_order, gt_map


def summarize_one_file(dataset: str, csv_path: Path, gt_order, gt_map, timeout_s: int):
    params = _parse_params_from_name(dataset, csv_path.name)
    if not params:
        return None

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
        if dataset == "gsm8k":
            pred = _extract_last_number(ans_s)
            exp = _extract_last_number(gt_obj.get("answer"))
            if pred is not None and exp is not None:
                try:
                    ok = abs(float(pred) - float(exp)) < 1e-3
                except Exception:
                    ok = False

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
            ok = _sandbox_eval(user_code, test_code, timeout_s)

        elif dataset == "mbpp":
            code = _extract_code(ans_s)
            test_code = _mbpp_test_code(gt_obj)
            ok = _sandbox_eval(code + "\n", test_code, timeout_s)

        if ok:
            n_correct += 1

    acc = (n_correct / n_total) if n_total else 0.0
    cost_mean = statistics.fmean(costs) if costs else 0.0
    lat_mean = statistics.fmean(lats) if lats else 0.0

    return {
        **params,
        "acc": acc,
        "cost": cost_mean,
        "lat": lat_mean,
        "n_present": float(n_present),
        "n_total": float(n_total),
        "file": str(csv_path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp_root", required=True)
    ap.add_argument("--timeout_s", type=int, default=3)
    args = ap.parse_args()

    exp_root = Path(args.exp_root).resolve()
    dataset = args.dataset

    dev_jsonl = exp_root / "dev10" / f"{dataset}_dev10.jsonl"
    grid_dir = exp_root / "dev_grid" / dataset
    out_summary = exp_root / "dev_summary" / f"{dataset}_grid_summary.csv"
    out_best = exp_root / "best_cfg" / f"{dataset}_best.json"

    if not dev_jsonl.exists():
        raise SystemExit(f"Missing dev input: {dev_jsonl}")
    if not grid_dir.exists():
        raise SystemExit(f"Missing grid dir: {grid_dir}")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_best.parent.mkdir(parents=True, exist_ok=True)

    gt_order, gt_map = _load_gt(dataset, dev_jsonl)

    csv_files = sorted([p for p in grid_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise SystemExit(f"No CSV files under {grid_dir}")

    # If there are seed-tagged grid files, prefer them to avoid mixing legacy filenames.
    has_seed_tagged = any(re.search(r"_seed\d+\.csv$", p.name) for p in csv_files)
    if has_seed_tagged:
        csv_files = [p for p in csv_files if re.search(r"_seed\d+\.csv$", p.name)]

    rows = []
    for p in csv_files:
        r = summarize_one_file(dataset, p, gt_order, gt_map, int(args.timeout_s))
        if r:
            rows.append(r)

    if not rows:
        raise SystemExit(f"No valid grid runs found under {grid_dir}")

    rows.sort(key=lambda x: (-x["acc"], x["cost"], x["lat"]))

    with out_summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "t1",
                "t2",
                "wq",
                "wr",
                "wc",
                "mr",
                "acc_mean",
                "acc_std",
                "cost_mean",
                "cost_std",
                "latency_mean",
                "latency_std",
                "n_present_mean",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["t1"],
                    r["t2"],
                    r["wq"],
                    r["wr"],
                    r["wc"],
                    r["mr"],
                    f"{r['acc']:.12f}",
                    "0.0",
                    f"{r['cost']:.12f}",
                    "0.0",
                    f"{r['lat']:.12f}",
                    "0.0",
                    f"{r['n_present']:.1f}",
                ]
            )

    best = rows[0]
    best_payload = {
        "dataset": dataset,
        "tau1": best["t1"],
        "tau2": best["t2"],
        "w_q": best["wq"],
        "w_r": best["wr"],
        "w_c": best["wc"],
        "min_rep": best["mr"],
        "dev_acc": best["acc"],
        "dev_cost_mean": best["cost"],
        "dev_latency_mean": best["lat"],
        "dev_n_present": best["n_present"],
        "dev_n_total": best["n_total"],
        "grid_csv": best["file"],
    }

    with out_best.open("w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2)

    print(f"[OK] wrote {out_summary}")
    print(f"[OK] wrote {out_best}")
    print(
        f"[BEST] {dataset}: acc={best['acc']:.4f} cost={best['cost']:.6f} lat={best['lat']:.2f} t1={best['t1']} t2={best['t2']} wq={best['wq']} wr={best['wr']} wc={best['wc']} mr={best['mr']}"
    )


if __name__ == "__main__":
    main()
