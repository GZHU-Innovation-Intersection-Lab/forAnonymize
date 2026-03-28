#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src").is_dir() and (p / "configs").is_dir():
            return p
    return here.parents[3]


def _default_exp_root() -> Path:
    env = os.environ.get("EXP_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def _resolve_path_maybe_under_exp_root(path_str: str, exp_root: Path) -> str:
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(exp_root / p)


def _read_jsonl_task_ids(path: Path):
    ids = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            tid = obj.get("task_id")
            if tid is None:
                continue
            ids.append(str(tid))
    return ids


def _count_completed_ok_task_ids(csv_path: Path):
    if (not csv_path.exists()) or csv_path.stat().st_size <= 0:
        return set()
    last_row = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                tid = row.get("task_id")
                if tid is None:
                    continue
                last_row[str(tid)] = row
    except Exception:
        return set()

    completed = set()
    for tid, row in last_row.items():
        err = str(row.get("error") or "").strip()
        ans = str(row.get("answer") or "").strip()
        if (err == "") and (ans != ""):
            completed.add(tid)
    return completed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--exp_root", default="")
    ap.add_argument("--input_jsonl", default="")
    ap.add_argument("--best_cfg_dataset", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout_sec", type=int, default=24000)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--log_dir", default="")
    ap.add_argument("--clean_rep_state", action="store_true")
    ap.add_argument("--force", action="store_true")

    ap.add_argument("--max_k", type=int, default=3)
    ap.add_argument("--budget_usd", type=float, default=5.0)
    ap.add_argument("--eta", type=float, default=0.3)

    ap.add_argument("--variant_label", default="TrustRoute")
    ap.add_argument("--no_short_code_prompt", action="store_true")
    ap.add_argument("--no_light_tests", action="store_true")
    ap.add_argument(
        "--stage2_strategy",
        default="vote",
        choices=["vote", "direct_fallback", "max_conf", "strongest_queried"],
    )
    ap.add_argument("--fallback_candidate_name", default="")
    ap.add_argument(
        "--usal_mode",
        default="default",
        choices=["default", "uniform", "no_agreement", "task_only"],
    )
    return ap.parse_args()


def main():
    repo_root = _find_repo_root()
    args = parse_args()
    exp_root = Path(args.exp_root).resolve() if args.exp_root else _default_exp_root()

    best_cfg_dataset = args.best_cfg_dataset.strip() or args.dataset
    best_cfg_path = exp_root / "best_cfg" / f"{best_cfg_dataset}_best.json"
    if not best_cfg_path.exists():
        raise SystemExit(f"Missing best_cfg: {best_cfg_path}")

    with best_cfg_path.open("r", encoding="utf-8") as f:
        best = json.load(f)

    tau1 = float(best["tau1"])
    tau2 = float(best["tau2"])
    wq = float(best["w_q"])
    wr = float(best["w_r"])
    wc = float(best["w_c"])
    mr = float(best.get("min_rep", 0.0))

    input_jsonl = (
        Path(_resolve_path_maybe_under_exp_root(args.input_jsonl, exp_root)).resolve()
        if args.input_jsonl
        else (exp_root / "test90" / f"{args.dataset}_test90.jsonl")
    )
    if not input_jsonl.exists():
        raise SystemExit(f"Missing test90 input: {input_jsonl}")

    out_dir = (
        Path(_resolve_path_maybe_under_exp_root(args.out_dir, exp_root)).resolve()
        if args.out_dir
        else (exp_root / "test90_runs_variant" / args.variant_label / args.dataset)
    )
    log_dir = (
        Path(_resolve_path_maybe_under_exp_root(args.log_dir, exp_root)).resolve()
        if args.log_dir
        else (exp_root / "logs" / "test90_variant" / args.variant_label / args.dataset)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"best_tau1_{tau1}_tau2_{tau2}_wq_{wq}_wr_{wr}_wc_{wc}_mr_{mr}_seed{args.seed}"
    )
    h = hashlib.sha1(f"{args.variant_label}|{run_id}".encode("utf-8")).hexdigest()[:10]
    rep_tag = f"{args.dataset}_{args.variant_label}_{h}"

    out_csv = out_dir / f"{args.dataset}_{args.variant_label}_{run_id}.csv"
    log_file = log_dir / f"{args.dataset}_{args.variant_label}_{run_id}.log"

    if (not args.force) and out_csv.exists():
        test_ids = _read_jsonl_task_ids(input_jsonl)
        done_ids = _count_completed_ok_task_ids(out_csv)
        n_total = len(test_ids)
        n_done = len(set(test_ids).intersection(done_ids))
        if (n_total > 0) and (n_done >= n_total):
            print(f"[SKIP] already complete ({n_done}/{n_total}): {out_csv}")
            return
        if n_total > 0:
            print(f"[RESUME] incomplete ({n_done}/{n_total}), continue: {out_csv}")
    else:
        n_total = 0
        n_done = 0

    should_clean_rep_state = args.clean_rep_state and not (
        (not args.force) and out_csv.exists() and n_total > 0 and n_done < n_total
    )

    if should_clean_rep_state:
        cache_dir = repo_root / "ISSTA" / "cache"
        rep_fp = cache_dir / f"rep_state_{rep_tag}_seed{args.seed}.json"
        try:
            if rep_fp.exists():
                rep_fp.unlink()
        except Exception:
            pass

    run_exp_py = exp_root / "scripts" / "run_exp.py"
    if not run_exp_py.exists():
        raise SystemExit(f"Missing run_exp.py: {run_exp_py}")

    cmd = [
        "python",
        "-u",
        str(run_exp_py),
        "--method",
        args.variant_label,
        "--input",
        str(input_jsonl),
        "--seed",
        str(args.seed),
        "--tau1",
        str(tau1),
        "--tau2",
        str(tau2),
        "--max_k",
        str(int(args.max_k)),
        "--budget_usd",
        str(float(args.budget_usd)),
        "--eta",
        str(float(args.eta)),
        "--w_q",
        str(wq),
        "--w_r",
        str(wr),
        "--w_c",
        str(wc),
        "--min_rep",
        str(mr),
        "--out",
        str(out_csv),
        "--log_file",
        str(log_file),
    ]

    if args.no_short_code_prompt:
        cmd.append("--no_short_code_prompt")
    if args.no_light_tests:
        cmd.append("--no_light_tests")
    if args.stage2_strategy != "vote":
        cmd += ["--stage2_strategy", args.stage2_strategy]
    if args.fallback_candidate_name:
        cmd += ["--fallback_candidate_name", args.fallback_candidate_name]
    if args.usal_mode != "default":
        cmd += ["--usal_mode", args.usal_mode]
    if args.limit and int(args.limit) > 0:
        cmd += ["--limit", str(int(args.limit))]

    env = os.environ.copy()
    env["ISSTA_REP_TAG"] = rep_tag
    env.setdefault("ISSTA_TEE_STDOUT", "0")

    print(f"[RUN] {args.dataset} {args.variant_label} -> {out_csv}")
    print(f"[LOG] {log_file}")
    print(f"[CMD] {' '.join(cmd)}")
    subprocess.run(
        cmd,
        check=True,
        timeout=int(args.timeout_sec),
        env=env,
    )
    print(f"[OK] wrote {out_csv}")


if __name__ == "__main__":
    main()
