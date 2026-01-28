import argparse
import hashlib
import itertools
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--max_workers", type=int, default=1)
    p.add_argument("--clean_rep_state", action="store_true")

    p.add_argument(
        "--rep_state_scope",
        choices=["per_run", "shared"],
        default="per_run",
        help="Reputation state isolation. per_run: each hyperparam config has its own rep_state file; shared: all configs share one.",
    )

    p.add_argument("--timeout_sec", type=int, default=9000)

    return p.parse_args()


TAU1_LIST = [0.75, 0.85, 0.95]
TAU2_LIST = [0.67, 1.0]
MIN_REP_LIST = [0.0]

MAX_K = 3
BUDGET_USD = 5.0
ETA = 0.3

WEIGHT_PRESETS = [
    (0.40, 0.30, 0.30),
    (0.60, 0.25, 0.15),
    (0.25, 0.60, 0.15),
    (0.25, 0.25, 0.50),
]

EXTRA_RUNS = [
    (0.75, 0.5, (0.40, 0.30, 0.30), 0.0),
    (0.95, 0.8, (1.0, 0.3, 0.2), 0.0),
]


def run_experiment(repo_root: Path, exp_root: Path, args, params):
    t1, t2, (wq, wr, wc), min_rep = params

    run_id = f"tau1_{t1}_tau2_{t2}_wq_{wq}_wr_{wr}_wc_{wc}_mr_{min_rep}_seed{args.seed}"

    rep_tag = ""
    if str(getattr(args, "rep_state_scope", "per_run")) == "per_run":
        h = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:10]
        rep_tag = f"{args.dataset}_TrustRoute_{h}"
    else:
        rep_tag = f"{args.dataset}_TrustRoute"

    out_csv = Path(args.out_dir) / f"{args.dataset}_TrustRoute_{run_id}.csv"
    log_file = Path(args.log_dir) / f"{args.dataset}_TrustRoute_{run_id}.log"

    if out_csv.exists() and out_csv.stat().st_size > 100:
        return f"Skip: {run_id}"

    os.makedirs(out_csv.parent, exist_ok=True)
    os.makedirs(log_file.parent, exist_ok=True)

    if getattr(args, "clean_rep_state", False):
        cache_dir = repo_root / "ISSTA" / "cache"
        # For TrustRoute, lite_utils uses ISSTA_REP_TAG + ISSTA_SEED to derive the file name.
        # We delete only the tag-specific file for this run to avoid affecting other runs.
        rep_fp = cache_dir / f"rep_state_{rep_tag}_seed{args.seed}.json"
        try:
            if rep_fp.exists():
                rep_fp.unlink()
        except Exception:
            pass

    run_exp_py = exp_root / "scripts" / "run_exp.py"

    cmd = [
        "python",
        "-u",
        str(run_exp_py),
        "--method",
        "TrustRoute",
        "--input",
        args.input,
        "--seed",
        str(args.seed),
        "--tau1",
        str(t1),
        "--tau2",
        str(t2),
        "--max_k",
        str(MAX_K),
        "--budget_usd",
        str(BUDGET_USD),
        "--eta",
        str(ETA),
        "--w_q",
        str(wq),
        "--w_r",
        str(wr),
        "--w_c",
        str(wc),
        "--min_rep",
        str(min_rep),
        "--out",
        str(out_csv),
    ]

    if args.limit and int(args.limit) > 0:
        cmd += ["--limit", str(args.limit)]

    with open(log_file, "w", encoding="utf-8") as f_log:
        try:
            subprocess.run(
                cmd,
                stdout=f_log,
                stderr=f_log,
                check=True,
                timeout=int(args.timeout_sec),
                cwd=str(repo_root),
                env={
                    **os.environ,
                    "EXP_ROOT": str(exp_root),
                    "ISSTA_SEED": str(args.seed),
                    "ISSTA_REP_TAG": rep_tag,
                },
            )
            return f"Done: {run_id}"
        except subprocess.TimeoutExpired:
            return f"Timeout: {run_id}"
        except subprocess.CalledProcessError:
            return f"Failed: {run_id}"


def main():
    repo_root = _find_repo_root()
    exp_root = _default_exp_root()

    args = parse_args()

    args.input = _resolve_path_maybe_under_exp_root(args.input, exp_root)

    if not args.out_dir:
        args.out_dir = str(exp_root / "dev_grid" / args.dataset)
    else:
        args.out_dir = _resolve_path_maybe_under_exp_root(args.out_dir, exp_root)

    if not args.log_dir:
        args.log_dir = str(exp_root / "logs" / "dev_grid" / args.dataset)
    else:
        args.log_dir = _resolve_path_maybe_under_exp_root(args.log_dir, exp_root)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    combos = list(itertools.product(TAU1_LIST, TAU2_LIST, WEIGHT_PRESETS, MIN_REP_LIST))

    for cfg in EXTRA_RUNS:
        if cfg not in combos:
            combos.append(cfg)

    print(
        f"Grid: dataset={args.dataset} seed={args.seed} n={len(combos)} out_dir={args.out_dir} log_dir={args.log_dir} rep_state_scope={args.rep_state_scope} max_workers={args.max_workers} clean_rep_state={bool(args.clean_rep_state)}",
        flush=True,
    )

    started = time.time()

    max_workers = max(1, int(args.max_workers))
    if max_workers == 1:
        for i, params in enumerate(combos, start=1):
            res = run_experiment(repo_root, exp_root, args, params)
            print(f"[{i}/{len(combos)}] {res}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(run_experiment, repo_root, exp_root, args, params) for params in combos]
            for i, fut in enumerate(as_completed(futs), start=1):
                print(f"[{i}/{len(combos)}] {fut.result()}", flush=True)

    print(f"All done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
