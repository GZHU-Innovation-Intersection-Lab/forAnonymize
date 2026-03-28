# src/tasks/loader_real.py
import json
import os
from typing import Any, Dict, List, Optional

def _ensure_hf_mirror():
    # 仅当未显式设置时，默认使用 HF 镜像
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

def _pick_first_exist(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def _extract_last_number(s: str) -> Optional[str]:
    import re
    if not isinstance(s, str):
        s = str(s)
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else None

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out

def _load_gsm8k_local(p: str) -> List[Dict[str, Any]]:
    items = _read_jsonl(p)
    tasks: List[Dict[str, Any]] = []
    for i, r in enumerate(items):
        q = r.get("question") or r.get("prompt") or r.get("input") or ""
        a = r.get("answer") or r.get("reference") or r.get("target") or ""
        ref_num = _extract_last_number(a) or (a.strip() if isinstance(a, str) else "")
        tid = r.get("id") or r.get("name") or f"gsm8k-{i+1}"
        tasks.append({
            "id": tid,
            "name": tid,
            "dataset": "gsm8k",
            "question": q,
            "prompt": f"Q: {q}\nA:",
            "input": q,
            "reference": str(ref_num),
        })
    return tasks

def _load_humaneval_local(p: str) -> List[Dict[str, Any]]:
    items = _read_jsonl(p)
    tasks: List[Dict[str, Any]] = []
    for r in items:
        tid = r.get("task_id") or r.get("id") or r.get("name")
        if not tid:
            continue
        prompt = r.get("prompt") or ""
        ref = r.get("canonical_solution") or r.get("reference") or ""
        # 兼容 openai/human-eval 官方 JSONL（HumanEval.jsonl）
        if not prompt and "prompt" in r:
            prompt = r["prompt"]
        tasks.append({
            "id": tid,
            "name": tid,
            "dataset": "humaneval",
            "prompt": prompt,
            "question": f"Implement the function as required:\n{prompt}",
            "input": "",
            "reference": ref,
        })
    return tasks

def _load_mbpp_local(p: str) -> List[Dict[str, Any]]:
    items = _read_jsonl(p)
    tasks: List[Dict[str, Any]] = []
    for r in items:
        tid = r.get("task_id") or r.get("id") or r.get("name")
        if not tid:
            # 兼容 google-research/mbpp 的结构：没有 task_id/name 时用序号
            continue
        prompt = r.get("prompt") or r.get("text") or ""
        ref = r.get("code") or r.get("reference") or ""
        tasks.append({
            "id": f"mbpp-{tid}",
            "name": f"mbpp-{tid}",
            "dataset": "mbpp",
            "prompt": prompt,
            "question": prompt,
            "input": "",
            "reference": ref,
        })
    return tasks

def _load_hf_gsm8k(limit: int) -> List[Dict[str, Any]]:
    _ensure_hf_mirror()
    try:
        from datasets import load_dataset
        ds = None
        try:
            ds = load_dataset("gsm8k", "main", split="test")
        except Exception:
            ds = load_dataset("gsm8k", split="test")
        items = []
        for i, r in enumerate(ds):
            q = r.get("question") or ""
            a = r.get("answer") or ""
            ref_num = _extract_last_number(a) or ""
            tid = r.get("id") or f"gsm8k-{i+1}"
            items.append({
                "id": tid, "name": tid, "dataset": "gsm8k",
                "question": q, "prompt": f"Q: {q}\nA:", "input": q, "reference": str(ref_num)
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception:
        return []

def _load_hf_humaneval(limit: int) -> List[Dict[str, Any]]:
    _ensure_hf_mirror()
    try:
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        items = []
        for i, r in enumerate(ds):
            tid = r.get("task_id")
            prompt = r.get("prompt") or ""
            ref = r.get("canonical_solution") or ""
            items.append({
                "id": tid, "name": tid, "dataset": "humaneval",
                "prompt": prompt,
                "question": f"Implement the function as required:\n{prompt}",
                "input": "", "reference": ref
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception:
        return []

def _load_hf_mbpp(limit: int) -> List[Dict[str, Any]]:
    _ensure_hf_mirror()
    try:
        from datasets import load_dataset
        ds = load_dataset("mbpp", split="test")
        items = []
        for i, r in enumerate(ds):
            tid = r.get("task_id") or r.get("id") or f"{i+1}"
            prompt = r.get("text") or r.get("prompt") or ""
            ref = r.get("code") or ""
            items.append({
                "id": f"mbpp-{tid}", "name": f"mbpp-{tid}", "dataset": "mbpp",
                "prompt": prompt, "question": prompt, "input": "", "reference": ref
            })
            if limit and len(items) >= limit:
                break
        return items
    except Exception:
        return []

def load_tasks(datasets: List[str], limit: int = 0, seeds: Optional[List[str]] = None, data_dir: str = "data") -> Dict[str, List[Dict[str, Any]]]:
    """
    data_dir 兼容以下本地路径结构：
    - GSM8K: {data_dir}/gsm8k_test.jsonl 或 {data_dir}/gsm8k/test.jsonl 或 benchmarks/gsm8k/test.jsonl
    - HumanEval: {data_dir}/humaneval/HumanEval.jsonl 或 {data_dir}/HumanEval.jsonl 或 benchmarks/humaneval/HumanEval.jsonl
    - MBPP: {data_dir}/mbpp.jsonl 或 {data_dir}/mbpp/mbpp.jsonl 或 benchmarks/mbpp/mbpp.jsonl
    本地不存在时，才尝试走 HF（已默认启用镜像）。
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for ds in datasets:
        name = (ds or "").strip().lower()

        if name == "gsm8k":
            local = _pick_first_exist([
                os.path.join(data_dir, "gsm8k_test.jsonl"),
                os.path.join(data_dir, "gsm8k", "test.jsonl"),
                os.path.join("benchmarks", "gsm8k", "test.jsonl"),
                os.path.join("benchmarks", "gsm8k", "test_200.jsonl"),
            ])
            items = _load_gsm8k_local(local) if local else _load_hf_gsm8k(limit)
            if limit and limit > 0:
                items = items[:limit]
            out["gsm8k"] = items

        elif name in {"humaneval", "human-eval", "human_eval"}:
            local = _pick_first_exist([
                os.path.join(data_dir, "humaneval", "HumanEval.jsonl"),
                os.path.join(data_dir, "HumanEval.jsonl"),
                os.path.join("benchmarks", "humaneval", "HumanEval.jsonl"),
            ])
            items = _load_humaneval_local(local) if local else _load_hf_humaneval(limit)
            if limit and limit > 0:
                items = items[:limit]
            out["humaneval"] = items

        elif name == "mbpp":
            local = _pick_first_exist([
                os.path.join(data_dir, "mbpp.jsonl"),
                os.path.join(data_dir, "mbpp", "mbpp.jsonl"),
                os.path.join("benchmarks", "mbpp", "mbpp.jsonl"),
            ])
            items = _load_mbpp_local(local) if local else _load_hf_mbpp(limit)
            if limit and limit > 0:
                items = items[:limit]
            out["mbpp"] = items

        else:
            out[name] = []
    return out