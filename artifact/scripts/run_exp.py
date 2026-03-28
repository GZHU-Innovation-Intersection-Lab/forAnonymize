"""
运行单个实验方法 (Fixed Version with Smart Resume & Correct Args)
"""
import argparse, csv, json, os, sys, asyncio
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
    p2 = exp_root / p
    return str(p2)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="输入数据文件")
    p.add_argument("--method", required=True, help="方法名称")
    p.add_argument("--out", required=True, help="输出CSV文件")
    p.add_argument("--seed", type=int, default=1, help="随机种子")
    p.add_argument("--limit", type=int, default=0, help="限制任务数（0=全部）")
    p.add_argument("--log_file", default="", help="可选：把 stdout/stderr 同时写入该日志文件")
    p.add_argument(
        "--candidate_name",
        default="",
        help="可选：只保留指定候选模型（匹配 name 或 model），用于 stronger single-model baseline",
    )

    p.add_argument("--tau1", type=float, default=0.95)
    p.add_argument("--tau2", type=float, default=0.80)
    p.add_argument("--max_k", type=int, default=3)
    p.add_argument("--budget_usd", type=float, default=5.0)
    p.add_argument("--eta", type=float, default=0.3)
    p.add_argument("--max_retries", type=int, default=1)

    p.add_argument("--w_q", type=float, default=1.0)
    p.add_argument("--w_r", type=float, default=0.3)
    p.add_argument("--w_c", type=float, default=0.2)
    p.add_argument("--min_rep", type=float, default=0.0)

    p.add_argument("--no_cost_ranking", action="store_true")
    p.add_argument("--no_reputation", action="store_true")
    p.add_argument("--no_diversity", action="store_true")
    p.add_argument("--no_short_code_prompt", action="store_true")
    p.add_argument("--no_light_tests", action="store_true")
    p.add_argument(
        "--stage2_strategy",
        default="vote",
        choices=["vote", "direct_fallback", "max_conf", "strongest_queried"],
    )
    p.add_argument("--fallback_candidate_name", default="")
    p.add_argument(
        "--usal_mode",
        default="default",
        choices=["default", "uniform", "no_agreement", "task_only"],
    )

    return p.parse_args()


def main():
    repo_root = _find_repo_root()
    sys.path.insert(0, str(repo_root))

    from src.agents.pool import build_agents
    from src.runner.executor import run_method
    from src.runner.lite_utils import reset_rep_state

    exp_root = _default_exp_root()

    args = parse_args()

    if not os.path.exists(args.input):
        args.input = _resolve_path_maybe_under_exp_root(args.input, exp_root)

    args.out = _resolve_path_maybe_under_exp_root(args.out, exp_root)

    if args.log_file:
        args.log_file = _resolve_path_maybe_under_exp_root(args.log_file, exp_root)

    os.environ["ISSTA_SEED"] = str(args.seed)

    resume_mode = os.path.exists(args.out)

    if "TrustRoute" in args.method or "Ours" in args.method:
        os.environ["ISSTA_SEED"] = str(args.seed)
        if not resume_mode:
            if not os.path.exists(args.out):
                reset_rep_state()

    ds_name = "unknown"
    if "gsm" in args.input.lower():
        ds_name = "gsm8k"
    elif "mbpp" in args.input.lower():
        ds_name = "mbpp"
    elif "human" in args.input.lower():
        ds_name = "humaneval"
    elif "hellaswag" in args.input.lower():
        ds_name = "hellaswag"
    elif "arc_challenge" in args.input.lower() or "arc" in args.input.lower():
        ds_name = "arc_challenge"
    elif "winogrande" in args.input.lower():
        ds_name = "winogrande"
    elif "mmlu" in args.input.lower():
        ds_name = "mmlu"

    if "TrustRoute" in args.method or "Ours" in args.method:
        if not os.environ.get("ISSTA_REP_TAG", "").strip():
            os.environ["ISSTA_REP_TAG"] = f"{ds_name}_{args.method}"
        else:
            print(f"[RepState] Using ISSTA_REP_TAG from env: {os.environ.get('ISSTA_REP_TAG')}")

    log_fp = None
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        log_fp = open(args.log_file, "a", encoding="utf-8", buffering=1)
        # Background reruns can deadlock when they inherit an unread stdout pipe.
        # Default to log-only mode unless explicitly asked to mirror to stdout.
        tee_stdout = os.environ.get("ISSTA_TEE_STDOUT", "0").strip() == "1"
        if tee_stdout:
            sys.stdout = _Tee(log_fp, sys.stdout)
            sys.stderr = _Tee(log_fp, sys.stderr)
        else:
            sys.stdout = _Tee(log_fp)
            sys.stderr = _Tee(log_fp)

    try:
        print(f"🚀 Running {args.method} on {ds_name}")
        print(f"   Input: {args.input}")
        print(f"   Output: {args.out}")
        print(f"   Seed: {args.seed}")

        tasks = []
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            return

        with open(args.input, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON at line {i+1}: {e}")
                    continue

                tid = obj.get("task_id") or obj.get("id") or f"{ds_name}-{i}"
                q = obj.get("prompt") or obj.get("question") or obj.get("input") or ""

                tasks.append(
                    {
                        "dataset": ds_name,
                        "task_id": tid,
                        "prompt": q,
                        "test": obj.get("test") or obj.get("tests") or "",
                        "entry_point": obj.get("entry_point") or "",
                        "test_list": obj.get("test_list"),
                        "assertion": obj.get("assertion") or "",
                        "contract": obj.get("contract") or "",
                        "answer": obj.get("answer", ""),
                    }
                )

        print(f"✅ Loaded {len(tasks)} raw tasks")

        if os.path.exists(args.out):
            last_by_task = {}
            try:
                with open(args.out, "r", encoding="utf-8", newline="") as f_csv:
                    reader = csv.DictReader(f_csv)
                    for row in reader:
                        task_id = str(row.get("task_id", "")).strip()
                        if task_id:
                            last_by_task[task_id] = row
            except Exception as e:
                print(f"⚠️  Failed to read existing output for resume: {e}")
                last_by_task = {}

            if last_by_task:
                completed_ids = {
                    task_id
                    for task_id, row in last_by_task.items()
                    if str(row.get("error", "")).strip() == ""
                    and str(row.get("answer", "")).strip() != ""
                }

                before = len(tasks)
                tasks = [t for t in tasks if str(t.get("task_id")) not in completed_ids]
                print(
                    f"Resume: {len(completed_ids)} completed ok, remaining {len(tasks)} (was {before})"
                )

        if args.limit > 0:
            tasks = tasks[: args.limit]
            print(f"   Limit applied: {len(tasks)} tasks")

        print()

        if not tasks:
            print("🎉 All tasks completed! Nothing to run.")
            return

        config_path = repo_root / "configs" / "agents.yaml"

        try:
            agents = build_agents(str(config_path))
        except Exception as e:
            print(f"❌ Config Error: {e}")
            return

        cands = [a for a in agents if a.extra.get("role") == "candidate"]
        if args.candidate_name.strip():
            needle = args.candidate_name.strip()
            cands = [
                a for a in cands
                if a.name == needle or a.model == needle
            ]
            if not cands:
                print(f"❌ candidate_name not found in candidate pool: {needle}")
                return
            print(f"✅ Candidate filter active: {needle} -> {len(cands)} candidate(s)")

        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        header = [
            "dataset",
            "task_id",
            "method",
            "seed",
            "answer",
            "cost_usd",
            "latency_s",
            "prompt_tokens",
            "completion_tokens",
            "agent_used",
            "reason",
            "error",
        ]

        write_header = not os.path.exists(args.out)

        print(f"{'='*80}")
        print("🏁 Starting execution loop...")
        print(f"{'='*80}\n")

        with open(args.out, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()

            for i, task in enumerate(tasks):
                print(f"[{i+1}/{len(tasks)}] Processing: {task['task_id']}", end="")

                row = {
                    "dataset": task["dataset"],
                    "task_id": task["task_id"],
                    "method": args.method,
                    "seed": args.seed,
                    "error": "",
                }

                try:
                    res = asyncio.run(
                        run_method(
                            method=args.method,
                            task=task,
                            candidates=cands,
                            rep_state={},
                            router=None,
                            budget_usd=args.budget_usd,
                            tau=0.6,
                            judge=None,
                            args=args,
                        )
                    )

                    row.update(
                        {
                            "answer": res.get("candidate", "") if res else "",
                            "cost_usd": res.get("cost_usd", 0) if res else 0,
                            "latency_s": res.get("latency_s", 0) if res else 0,
                            "prompt_tokens": res.get("prompt_tokens", 0) if res else 0,
                            "completion_tokens": res.get("completion_tokens", 0)
                            if res
                            else 0,
                            "agent_used": res.get("agent_used", "") if res else "",
                            "reason": res.get("reason", "") if res else "",
                        }
                    )

                    if res:
                        print(
                            f" ✅ ${res.get('cost_usd', 0):.4f} {res.get('latency_s', 0):.1f}s"
                        )
                    else:
                        print(" ⚠️  None result")

                except Exception as e:
                    row["error"] = str(e)
                    print(f" ❌ {str(e)[:100]}")

                writer.writerow(row)
                f.flush()

        print("\n✅ Batch completed.")

    finally:
        if log_fp is not None:
            try:
                log_fp.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
