# src/runner/executor.py - 包含所有实验方法

import asyncio
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.agents.base import BaseAgent, AgentResult
from src.runner.lite_utils import (
    load_rep_state,
    save_rep_state,
    update_reputation,
    get_agent_reputation,
    rank_agents_by_rep_cost as rank_agent_dicts_by_rep_cost,
)

def rank_agents_by_rep_cost(
    agents: List[BaseAgent],
    w_q: float = 1.0,
    w_r: float = 0.3,
    w_c: float = 0.2,
    min_rep: float = 0.0
) -> List[BaseAgent]:
    """根据质量、信誉和成本对代理进行排名"""
    scored = []
    for agent in agents:
        model_id = agent.model
        rep = get_agent_reputation(model_id)
        
        if rep < min_rep:
            continue
        
        cost_norm = agent.cpk / 0.1
        quality = agent.extra.get("quality", 0.7)
        score = w_q * quality + w_r * rep - w_c * cost_norm
        
        scored.append((score, agent))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [agent for _, agent in scored]


def _safe_price(obj, primary_key, aliases, default):
    keys = [primary_key] + list(aliases)
    for key in keys:
        val = getattr(obj, key, None)
        if val is not None:
            return float(val)
    if hasattr(obj, "extra") and isinstance(obj.extra, dict):
        for key in keys:
            val = obj.extra.get(key)
            if val is not None:
                return float(val)
    return default


def _build_agent_dicts(candidates: List[BaseAgent]) -> List[Dict[str, Any]]:
    agent_dicts = []
    for agent in candidates:
        p_in = _safe_price(agent, "prompt_price_per_1k", ("ppk",), 0.00015)
        p_out = _safe_price(agent, "completion_price_per_1k", ("cpk",), 0.00060)
        tier = getattr(agent, "extra", {}).get("tier", "unknown")
        agent_dicts.append(
            {
                "name": agent.name,
                "model": agent.model,
                "obj": agent,
                "tier": tier,
                "meta": {"pricing": {"input": p_in, "output": p_out}},
            }
        )
    return agent_dicts


async def _generate_from_agent_dict(
    agent_dict: Dict[str, Any],
    task: Dict[str, Any],
    prompt_suffix: str = "",
    temperature: float = 0.2,
    prompt_override: Optional[str] = None,
) -> Dict[str, Any]:
    agent_obj = agent_dict["obj"]
    prompt = prompt_override if prompt_override is not None else (task.get("prompt") or task.get("question") or str(task)) + prompt_suffix
    result = await agent_obj.generate(prompt, temperature=temperature)
    return {
        "text": result.text,
        "cost_usd": result.cost_usd,
        "latency_s": result.latency_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def _prompt_suffix_for_task(task: Dict[str, Any], args: Any) -> str:
    dataset = (task.get("dataset") or "").lower()
    if dataset in {"mbpp", "humaneval"} and not getattr(args, "no_short_code_prompt", False):
        return "\n\nPlease provide a concise solution with minimal comments."
    return ""


def _agent_strength_key(agent: Dict[str, Any]) -> tuple[float, float]:
    agent_obj = agent.get("obj")
    extra = getattr(agent_obj, "extra", {}) if agent_obj is not None else {}
    quality = float(extra.get("quality", 0.7)) if isinstance(extra, dict) else 0.7
    pricing = agent.get("meta", {}).get("pricing", {})
    avg_cost = float(pricing.get("input", 0.0)) + float(pricing.get("output", 0.0))
    return (quality, avg_cost)


def _rank_agent_dicts_for_task(agent_dicts: List[Dict[str, Any]], args: Any) -> List[Dict[str, Any]]:
    rep_state = load_rep_state()
    return rank_agent_dicts_by_rep_cost(
        agent_dicts,
        rep_state,
        w_q=getattr(args, "w_q", 1.0),
        w_r=getattr(args, "w_r", 0.3),
        w_c=getattr(args, "w_c", 0.2),
        min_rep=getattr(args, "min_rep", 0.0),
    )


_SMOOTHIE_STATE_CACHE: Dict[tuple, Dict[str, Any]] = {}
_CP_ROUTER_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _exp_root_path() -> Path:
    env = os.environ.get("EXP_ROOT", "").strip()
    if env:
        return Path(env).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    default = repo_root / "11111A_rerun_adj_para"
    return default if default.exists() else repo_root


def _task_prompt(task: Dict[str, Any]) -> str:
    return (
        task.get("prompt")
        or task.get("question")
        or task.get("input")
        or ""
    ).strip()


def _extract_last_number(text: str) -> Optional[str]:
    if text is None:
        return None
    s = str(text).replace(",", "")
    m = re.findall(r"####\s*([-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", s)
    if m:
        return str(m[-1]).strip()
    m = re.findall(r"[-+]?\d+(?:\.\d+)?", s)
    return str(m[-1]).strip() if m else None


def _parse_number(x: Any) -> Optional[float]:
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


def _extract_mcq_letter(text: str, dataset: str) -> Optional[str]:
    allow = "AB" if dataset == "winogrande" else "ABCD"
    t = ("" if text is None else str(text)).strip().upper()
    m = re.search(r"ANSWER\s*[:：=]\s*([%s])" % allow, t)
    if m:
        return m.group(1)
    m = re.findall(r"(?:^|\b)([%s])(?:\b|$)" % allow, t)
    return m[-1] if m else None


def _normalize_answer_for_dataset(text: str, dataset: str) -> Optional[str]:
    dataset = (dataset or "").lower()
    if dataset in {"arc_challenge", "hellaswag", "mmlu", "winogrande"}:
        return _extract_mcq_letter(text, dataset)
    if dataset == "gsm8k":
        num = _parse_number(_extract_last_number(text))
        return None if num is None else f"{num:.8f}"
    s = ("" if text is None else str(text)).strip()
    return s if s else None


def _tokenize_prompt(prompt: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (prompt or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0
    union = len(a | b)
    return inter / max(1, union)


def _read_jsonl_tasks(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                items.append(json.loads(s))
            except Exception:
                continue
    return items


def _read_last_rows_by_task_id(csv_path: Path) -> Dict[str, Dict[str, Any]]:
    import csv

    last: Dict[str, Dict[str, Any]] = {}
    if (not csv_path.exists()) or csv_path.stat().st_size <= 0:
        return last
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get("task_id") or "").strip()
            if tid:
                last[tid] = row
    return last


def _candidate_name_from_single_csv(path: Path) -> str:
    m = re.search(r"_SA-single_(.+?)_seed\d+\.csv$", path.name)
    if m:
        return m.group(1)
    return path.stem


def _best_single_candidate_name(dataset: str, candidates: List[BaseAgent]) -> Optional[str]:
    exp_root = _exp_root_path()
    best_json = exp_root / "dev10_summary_single" / f"{dataset}_dev10_single_model_best.json"
    if best_json.exists():
        try:
            obj = json.loads(best_json.read_text(encoding="utf-8"))
            name = str(obj.get("candidate_name") or "").strip()
            if name:
                return name
        except Exception:
            pass
    if not candidates:
        return None
    return max(candidates, key=lambda a: (float(a.extra.get("quality", 0.7)), -(a.ppk + a.cpk))).name


def _find_candidate_by_name(candidates: List[BaseAgent], name: str) -> Optional[BaseAgent]:
    for agent in candidates:
        if agent.name == name or agent.model == name:
            return agent
    return None


def _load_dev10_single_outputs(dataset: str, seed: int) -> Dict[str, Dict[str, str]]:
    exp_root = _exp_root_path()
    run_dir = exp_root / "dev10_runs_single" / dataset
    outputs: Dict[str, Dict[str, str]] = {}
    if not run_dir.exists():
        return outputs
    for csv_path in sorted(run_dir.glob(f"{dataset}_SA-single_*_seed{seed}.csv")):
        cand_name = _candidate_name_from_single_csv(csv_path)
        rows = _read_last_rows_by_task_id(csv_path)
        outputs[cand_name] = {
            tid: str(row.get("answer") or "")
            for tid, row in rows.items()
            if str(row.get("error") or "").strip() == "" and str(row.get("answer") or "").strip() != ""
        }
    return outputs


def _build_smoothie_state(
    dataset: str,
    seed: int,
    candidates: List[BaseAgent],
) -> Dict[str, Any]:
    key = (dataset, int(seed), tuple(sorted(a.name for a in candidates)))
    if key in _SMOOTHIE_STATE_CACHE:
        return _SMOOTHIE_STATE_CACHE[key]

    from .ours_lite_v2 import extract_and_repair_code, run_light_tests

    exp_root = _exp_root_path()
    dev_path = exp_root / "dev10" / f"{dataset}_dev10.jsonl"
    tasks = _read_jsonl_tasks(dev_path)
    outputs = _load_dev10_single_outputs(dataset, seed)
    strongest_name = _best_single_candidate_name(dataset, candidates) or (candidates[0].name if candidates else "")

    examples = []
    totals = {a.name: 0.0 for a in candidates}
    counts = {a.name: 0 for a in candidates}

    for task in tasks:
        task_id = str(task.get("task_id") or task.get("id") or "")
        if not task_id:
            continue
        prompt_tokens = _tokenize_prompt(_task_prompt(task))
        if not prompt_tokens:
            continue

        model_scores: Dict[str, float] = {}
        if dataset in {"mbpp", "humaneval"}:
            for agent in candidates:
                ans = outputs.get(agent.name, {}).get(task_id)
                if not ans:
                    continue
                code = extract_and_repair_code(ans)
                ok, msg = run_light_tests(code, task)
                model_scores[agent.name] = 1.0 if ok and msg == "pass" else 0.0
        else:
            normalized: Dict[str, str] = {}
            for agent in candidates:
                ans = outputs.get(agent.name, {}).get(task_id)
                norm = _normalize_answer_for_dataset(ans, dataset)
                if norm is not None:
                    normalized[agent.name] = norm
            if not normalized:
                continue
            freq = Counter(normalized.values())
            top_count = max(freq.values())
            top_answers = [ans for ans, cnt in freq.items() if cnt == top_count]
            if len(top_answers) == 1:
                pseudo = top_answers[0]
            else:
                pseudo = normalized.get(strongest_name)
                if pseudo is None:
                    continue
            for name, norm in normalized.items():
                model_scores[name] = 1.0 if norm == pseudo else 0.0

        if not model_scores:
            continue

        for name, val in model_scores.items():
            totals[name] = totals.get(name, 0.0) + float(val)
            counts[name] = counts.get(name, 0) + 1

        examples.append(
            {
                "tokens": prompt_tokens,
                "scores": model_scores,
            }
        )

    global_scores = {
        name: (totals.get(name, 0.0) / counts[name]) if counts.get(name, 0) else 0.0
        for name in totals
    }
    state = {"examples": examples, "global_scores": global_scores}
    _SMOOTHIE_STATE_CACHE[key] = state
    return state


def _select_smoothie_agent(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    state: Dict[str, Any],
) -> BaseAgent:
    prompt_tokens = _tokenize_prompt(_task_prompt(task))
    global_scores = state.get("global_scores", {})
    examples = state.get("examples", [])
    best_agent = candidates[0]
    best_score = -1.0
    for agent in candidates:
        prior = float(global_scores.get(agent.name, 0.0))
        num = 2.0 * prior
        den = 2.0
        for ex in examples:
            sim = _jaccard(prompt_tokens, ex.get("tokens", set()))
            if sim <= 0.0:
                continue
            num += sim * float(ex.get("scores", {}).get(agent.name, prior))
            den += sim
        score = num / den if den > 0 else prior
        tie = (float(agent.extra.get("quality", 0.7)), -(agent.ppk + agent.cpk))
        best_tie = (float(best_agent.extra.get("quality", 0.7)), -(best_agent.ppk + best_agent.cpk))
        if (score > best_score + 1e-12) or (abs(score - best_score) <= 1e-12 and tie > best_tie):
            best_score = score
            best_agent = agent
    return best_agent


def _mcq_labels(dataset: str) -> List[str]:
    return list("AB") if dataset == "winogrande" else list("ABCD")


def _parse_prob_response(text: str, labels: List[str]) -> tuple[Optional[str], Dict[str, float]]:
    answer = None
    probs: Dict[str, float] = {}
    s = "" if text is None else str(text).strip()
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            answer = str(obj.get("answer") or "").strip().upper()[:1] or None
            raw_probs = obj.get("probs") or obj.get("probabilities") or {}
            if isinstance(raw_probs, dict):
                for lab in labels:
                    if lab in raw_probs:
                        probs[lab] = max(0.0, float(raw_probs[lab]))
        except Exception:
            pass
    if not probs:
        for lab in labels:
            m_lab = re.search(rf"{lab}\s*[:=]\s*([01](?:\.\d+)?)", s, flags=re.IGNORECASE)
            if m_lab:
                try:
                    probs[lab] = max(0.0, float(m_lab.group(1)))
                except Exception:
                    pass
    if answer is None:
        answer = _extract_mcq_letter(s, "winogrande" if labels == list("AB") else "mmlu")
    if probs:
        total = sum(probs.values())
        if total > 0:
            probs = {k: v / total for k, v in probs.items()}
        else:
            probs = {}
    return answer, probs


async def _query_mcq_probs(agent: BaseAgent, task: Dict[str, Any]) -> Dict[str, Any]:
    dataset = (task.get("dataset") or "").lower()
    labels = _mcq_labels(dataset)
    probs_example = ",".join([f'"{lab}":0.0' for lab in labels])
    prompt = (
        f"{_task_prompt(task).rstrip()}\n\n"
        f"Return only compact JSON of the form "
        f'{{"answer":"{labels[0]}","probs":{{{probs_example}}}}}. '
        "The probabilities must sum to 1 exactly."
    )
    res = await agent.generate(prompt, temperature=0.0)
    answer, probs = _parse_prob_response(res.text, labels)
    return {
        "answer": answer,
        "probs": probs,
        "text": res.text,
        "cost_usd": res.cost_usd,
        "latency_s": res.latency_s,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
    }


async def _get_cprouter_calibration(
    dataset: str,
    small_agent: BaseAgent,
) -> Dict[str, Any]:
    key = (dataset, small_agent.name, small_agent.model)
    if key in _CP_ROUTER_CACHE:
        return _CP_ROUTER_CACHE[key]

    exp_root = _exp_root_path()
    cache_dir = exp_root / "cache" / "cprouter"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset}_{small_agent.name}.json"
    lock_path = cache_dir / f"{dataset}_{small_agent.name}.lock"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _CP_ROUTER_CACHE[key] = cached
            return cached
        except Exception:
            pass

    if dataset not in {"arc_challenge", "hellaswag", "mmlu", "winogrande"}:
        calib = {"dataset": dataset, "mode": "unsupported", "qhat": 0.0, "alpha": 0.1}
        _CP_ROUTER_CACHE[key] = calib
        return calib

    while lock_path.exists() and not cache_path.exists():
        await asyncio.sleep(2.0)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _CP_ROUTER_CACHE[key] = cached
            return cached
        except Exception:
            pass

    have_lock = False
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        have_lock = True
    except FileExistsError:
        while lock_path.exists() and not cache_path.exists():
            await asyncio.sleep(2.0)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            _CP_ROUTER_CACHE[key] = cached
            return cached

    try:
        tasks = _read_jsonl_tasks(exp_root / "dev10" / f"{dataset}_dev10.jsonl")
        scores: List[float] = []
        for task in tasks:
            gold = str(task.get("answer") or "").strip().upper()
            if not gold:
                continue
            out = await _query_mcq_probs(small_agent, task)
            probs = out.get("probs") or {}
            p_gold = float(probs.get(gold, 0.0))
            scores.append(max(0.0, 1.0 - p_gold))

        alpha = 0.1
        if scores:
            ordered = sorted(scores)
            q_idx = max(0, min(len(ordered) - 1, math.ceil((len(ordered) + 1) * (1 - alpha)) - 1))
            qhat = float(ordered[q_idx])
        else:
            qhat = 1.0

        calib = {"dataset": dataset, "mode": "mcq", "qhat": qhat, "alpha": alpha, "n_calib": len(scores)}
        if have_lock:
            cache_path.write_text(json.dumps(calib, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if have_lock and lock_path.exists():
            try:
                lock_path.unlink()
            except Exception:
                pass
    _CP_ROUTER_CACHE[key] = calib
    return calib


def _parse_confidence_value(text: str) -> Optional[float]:
    import re

    matches = re.findall(r"CONFIDENCE\s*[:=]\s*([01](?:\.\d+)?)", str(text).upper())
    if matches:
        try:
            return max(0.0, min(1.0, float(matches[-1])))
        except Exception:
            return None

    nums = re.findall(r"([01](?:\.\d+)?)", str(text))
    if nums:
        try:
            return max(0.0, min(1.0, float(nums[-1])))
        except Exception:
            return None
    return None


async def run_code_exec_rerank(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any,
) -> Dict[str, Any]:
    from .ours_lite_v2 import extract_and_repair_code, run_light_tests, _compute_confidence

    dataset = (task.get("dataset") or "").lower()
    if dataset not in {"mbpp", "humaneval"}:
        raise ValueError("CodeExecRerank only supports mbpp/humaneval")

    start_time = time.time()
    agent_dicts = _build_agent_dicts(candidates)
    ranked = _rank_agent_dicts_for_task(agent_dicts, args) or agent_dicts
    selected = ranked[: max(1, min(int(getattr(args, "max_k", 3)), len(ranked)))]
    prompt_suffix = _prompt_suffix_for_task(task, args)

    coros = [_generate_from_agent_dict(agent, task, prompt_suffix, temperature=0.2) for agent in selected]
    results = await asyncio.gather(*coros, return_exceptions=True)

    rows = []
    total_cost = 0.0
    total_pt = 0
    total_ct = 0
    for agent, res in zip(selected, results):
        if isinstance(res, Exception):
            continue
        total_cost += float(res.get("cost_usd", 0.0))
        total_pt += int(res.get("prompt_tokens", 0) or 0)
        total_ct += int(res.get("completion_tokens", 0) or 0)
        code = extract_and_repair_code(res.get("text", ""))
        ok, msg = run_light_tests(code, task)
        tests_ok = ok and msg == "pass"
        rows.append(
            {
                "agent": agent["name"],
                "text": res.get("text", ""),
                "tests_ok": tests_ok,
                "confidence": _compute_confidence(code, dataset, tests_ok),
                "strength": _agent_strength_key(agent),
            }
        )

    if not rows:
        return {
            "candidate": "",
            "cost_usd": total_cost,
            "latency_s": time.time() - start_time,
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "agent_used": "",
            "reason": "execution_rerank_no_result",
            "error": "no_results",
        }

    passed = [r for r in rows if r["tests_ok"]]
    if passed:
        best = max(passed, key=lambda r: (r["confidence"], r["strength"]))
        reason = "execution_rerank_pass"
    else:
        best = max(rows, key=lambda r: (r["confidence"], r["strength"]))
        reason = "execution_rerank_conf"

    return {
        "candidate": best["text"],
        "cost_usd": total_cost,
        "latency_s": time.time() - start_time,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "agent_used": best["agent"],
        "reason": reason,
    }


async def run_code_self_repair(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any,
) -> Dict[str, Any]:
    from .ours_lite_v2 import extract_and_repair_code, run_light_tests, _compute_confidence

    dataset = (task.get("dataset") or "").lower()
    if dataset not in {"mbpp", "humaneval"}:
        raise ValueError("CodeSelfRepair only supports mbpp/humaneval")

    start_time = time.time()
    agent_dicts = _build_agent_dicts(candidates)
    strongest = max(agent_dicts, key=_agent_strength_key)
    prompt_suffix = _prompt_suffix_for_task(task, args)

    first = await _generate_from_agent_dict(strongest, task, prompt_suffix, temperature=0.2)
    total_cost = float(first.get("cost_usd", 0.0))
    total_pt = int(first.get("prompt_tokens", 0) or 0)
    total_ct = int(first.get("completion_tokens", 0) or 0)

    code1 = extract_and_repair_code(first.get("text", ""))
    ok1, msg1 = run_light_tests(code1, task)
    tests_ok1 = ok1 and msg1 == "pass"
    conf1 = _compute_confidence(code1, dataset, tests_ok1)
    if tests_ok1:
        return {
            "candidate": first.get("text", ""),
            "cost_usd": total_cost,
            "latency_s": time.time() - start_time,
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "agent_used": strongest["name"],
            "reason": "self_repair_stage1_pass",
        }

    orig_prompt = task.get("prompt") or task.get("question") or ""
    repair_prompt = (
        f"{orig_prompt.rstrip()}\n\n"
        f"Your previous solution failed lightweight checks with status: {msg1}.\n"
        "Repair the solution and return only the corrected code.\n\n"
        f"Previous solution:\n```python\n{code1}\n```"
    )
    repaired = await _generate_from_agent_dict(strongest, task, "", temperature=0.2, prompt_override=repair_prompt)
    total_cost += float(repaired.get("cost_usd", 0.0))
    total_pt += int(repaired.get("prompt_tokens", 0) or 0)
    total_ct += int(repaired.get("completion_tokens", 0) or 0)

    code2 = extract_and_repair_code(repaired.get("text", ""))
    ok2, msg2 = run_light_tests(code2, task)
    tests_ok2 = ok2 and msg2 == "pass"
    conf2 = _compute_confidence(code2, dataset, tests_ok2)

    if tests_ok2 or conf2 >= conf1:
        chosen = repaired
        reason = "self_repair_repaired" if tests_ok2 else "self_repair_higher_conf"
    else:
        chosen = first
        reason = "self_repair_keep_original"

    return {
        "candidate": chosen.get("text", ""),
        "cost_usd": total_cost,
        "latency_s": time.time() - start_time,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "agent_used": strongest["name"],
        "reason": reason,
    }


async def run_automix_style(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any,
) -> Dict[str, Any]:
    from .ours_lite_v2 import extract_and_repair_code, run_light_tests, _compute_confidence

    start_time = time.time()
    dataset = (task.get("dataset") or "").lower()
    agent_dicts = _build_agent_dicts(candidates)
    ranked = _rank_agent_dicts_for_task(agent_dicts, args) or agent_dicts
    if not ranked:
        raise ValueError("No candidates for AutoMixStyle")

    top = ranked[0]
    prompt_suffix = _prompt_suffix_for_task(task, args)
    first = await _generate_from_agent_dict(top, task, prompt_suffix, temperature=0.2)
    total_cost = float(first.get("cost_usd", 0.0))
    total_pt = int(first.get("prompt_tokens", 0) or 0)
    total_ct = int(first.get("completion_tokens", 0) or 0)

    code1 = extract_and_repair_code(first.get("text", "")) if dataset in {"mbpp", "humaneval"} else first.get("text", "")
    tests_ok1 = False
    if dataset in {"mbpp", "humaneval"}:
        ok1, msg1 = run_light_tests(code1, task)
        tests_ok1 = ok1 and msg1 == "pass"
        if tests_ok1:
            return {
                "candidate": first.get("text", ""),
                "cost_usd": total_cost,
                "latency_s": time.time() - start_time,
                "prompt_tokens": total_pt,
                "completion_tokens": total_ct,
                "agent_used": top["name"],
                "reason": "automix_stage1_verified",
            }

    conf1 = _compute_confidence(code1, dataset, tests_ok1)
    orig_prompt = task.get("prompt") or task.get("question") or ""
    verify_prompt = (
        f"{orig_prompt.rstrip()}\n\n"
        f"Candidate answer:\n{first.get('text', '')}\n\n"
        "Estimate the probability that this candidate answer is correct. "
        "Reply with exactly one line: CONFIDENCE=<number between 0 and 1>."
    )
    verify = await _generate_from_agent_dict(top, task, "", temperature=0.0, prompt_override=verify_prompt)
    total_cost += float(verify.get("cost_usd", 0.0))
    total_pt += int(verify.get("prompt_tokens", 0) or 0)
    total_ct += int(verify.get("completion_tokens", 0) or 0)
    self_conf = _parse_confidence_value(verify.get("text", ""))
    combined_conf = conf1 if self_conf is None else max(conf1, self_conf)

    if combined_conf >= float(getattr(args, "tau1", 0.95)):
        return {
            "candidate": first.get("text", ""),
            "cost_usd": total_cost,
            "latency_s": time.time() - start_time,
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "agent_used": top["name"],
            "reason": "automix_self_verified",
        }

    remaining = [a for a in agent_dicts if a["name"] != top["name"]]
    strongest = max(remaining or agent_dicts, key=_agent_strength_key)
    second = await _generate_from_agent_dict(strongest, task, prompt_suffix, temperature=0.2)
    total_cost += float(second.get("cost_usd", 0.0))
    total_pt += int(second.get("prompt_tokens", 0) or 0)
    total_ct += int(second.get("completion_tokens", 0) or 0)

    code2 = extract_and_repair_code(second.get("text", "")) if dataset in {"mbpp", "humaneval"} else second.get("text", "")
    tests_ok2 = False
    if dataset in {"mbpp", "humaneval"}:
        ok2, msg2 = run_light_tests(code2, task)
        tests_ok2 = ok2 and msg2 == "pass"
    conf2 = _compute_confidence(code2, dataset, tests_ok2)

    choose_second = tests_ok2 or conf2 > combined_conf
    chosen = second if choose_second else first
    chosen_agent = strongest["name"] if choose_second else top["name"]
    reason = "automix_fallback_strongest" if choose_second else "automix_keep_stage1"

    return {
        "candidate": chosen.get("text", ""),
        "cost_usd": total_cost,
        "latency_s": time.time() - start_time,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "agent_used": chosen_agent,
        "reason": reason,
    }


async def run_smoothie(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any,
) -> Dict[str, Any]:
    start_time = time.time()
    dataset = (task.get("dataset") or "").lower()
    state = _build_smoothie_state(dataset, int(getattr(args, "seed", 1)), candidates)
    if not candidates:
        raise ValueError("No candidates for Smoothie")

    chosen = _select_smoothie_agent(task, candidates, state)
    agent_dict = _build_agent_dicts([chosen])[0]
    prompt_suffix = _prompt_suffix_for_task(task, args)
    res = await _generate_from_agent_dict(agent_dict, task, prompt_suffix, temperature=0.2)
    return {
        "candidate": res.get("text", ""),
        "cost_usd": float(res.get("cost_usd", 0.0)),
        "latency_s": time.time() - start_time,
        "prompt_tokens": int(res.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(res.get("completion_tokens", 0) or 0),
        "agent_used": chosen.name,
        "reason": "smoothie_local_pseudo_router",
    }


async def run_cp_router(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any,
) -> Dict[str, Any]:
    start_time = time.time()
    dataset = (task.get("dataset") or "").lower()
    if not candidates:
        raise ValueError("No candidates for CP-Router")

    small = min(candidates, key=lambda a: (a.ppk + a.cpk, -float(a.extra.get("quality", 0.7))))
    strongest_name = _best_single_candidate_name(dataset, candidates) or small.name
    strongest = _find_candidate_by_name(candidates, strongest_name) or max(
        candidates, key=lambda a: (float(a.extra.get("quality", 0.7)), -(a.ppk + a.cpk))
    )

    if dataset not in {"arc_challenge", "hellaswag", "mmlu", "winogrande"}:
        strongest_dict = _build_agent_dicts([strongest])[0]
        prompt_suffix = _prompt_suffix_for_task(task, args)
        res = await _generate_from_agent_dict(strongest_dict, task, prompt_suffix, temperature=0.2)
        return {
            "candidate": res.get("text", ""),
            "cost_usd": float(res.get("cost_usd", 0.0)),
            "latency_s": time.time() - start_time,
            "prompt_tokens": int(res.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(res.get("completion_tokens", 0) or 0),
            "agent_used": strongest.name,
            "reason": "cprouter_mcq_only_fallback",
        }

    calib = await _get_cprouter_calibration(dataset, small)
    first = await _query_mcq_probs(small, task)
    total_cost = float(first.get("cost_usd", 0.0))
    total_pt = int(first.get("prompt_tokens", 0) or 0)
    total_ct = int(first.get("completion_tokens", 0) or 0)

    labels = _mcq_labels(dataset)
    probs = first.get("probs") or {}
    qhat = float(calib.get("qhat", 1.0))
    prediction_set = [lab for lab in labels if (1.0 - float(probs.get(lab, 0.0))) <= qhat]
    pred_answer = str(first.get("answer") or "").strip().upper()
    if len(prediction_set) == 1 and pred_answer and pred_answer == prediction_set[0]:
        return {
            "candidate": pred_answer,
            "cost_usd": total_cost,
            "latency_s": time.time() - start_time,
            "prompt_tokens": total_pt,
            "completion_tokens": total_ct,
            "agent_used": small.name,
            "reason": "cprouter_prediction_set_size_1",
        }

    strong_dict = _build_agent_dicts([strongest])[0]
    prompt_suffix = _prompt_suffix_for_task(task, args)
    second = await _generate_from_agent_dict(strong_dict, task, prompt_suffix, temperature=0.2)
    total_cost += float(second.get("cost_usd", 0.0))
    total_pt += int(second.get("prompt_tokens", 0) or 0)
    total_ct += int(second.get("completion_tokens", 0) or 0)
    return {
        "candidate": second.get("text", ""),
        "cost_usd": total_cost,
        "latency_s": time.time() - start_time,
        "prompt_tokens": total_pt,
        "completion_tokens": total_ct,
        "agent_used": strongest.name,
        "reason": f"cprouter_escalate_setsize_{len(prediction_set)}",
    }


async def run_method(method: str, task: dict, candidates: list,
                     rep_state: dict, router, budget_usd: float, 
                     tau: float, judge, args) -> dict:
    """统一的方法调度器"""
    
    # TrustRoute系列
    if "TrustRoute" in method or "Ours" in method:
        from .ours_lite_v2 import run_ours_lite
        agent_dicts = _build_agent_dicts(candidates)
        
        # 🔧 修复：参数名改为 temperature
        async def gen_fn(agent_dict, task, suffix, temperature):
            res = await _generate_from_agent_dict(agent_dict, task, prompt_suffix=suffix, temperature=temperature)
            return {
                "text": res["text"],
                "cost_usd": res["cost_usd"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
            }
        
        async def judge_fn(task, text):
            return [{"score": 0.5}]
        
        no_rep = "NoRep" in method
        # Keep the two cost-related ablations distinct for future reruns:
        # - NoCost: preserve the ranking path but zero the cost weight.
        # - NoCostAware: bypass cost-aware ranking entirely.
        no_cost = ("NoCost" in method) and ("NoCostAware" not in method)
        no_cost_aware = "NoCostAware" in method
        no_parallel = "NoParallel" in method

        max_k = 1 if no_parallel else getattr(args, "max_k", 3)
        w_c = 0.0 if no_cost else getattr(args, "w_c", 0.2)
        
        return await run_ours_lite(
            task=task,
            candidate_agents=agent_dicts,
            generate_fn=gen_fn,
            judge_fn=judge_fn,
            budget_usd=getattr(args, "budget_usd", 5.0),
            tau1=getattr(args, "tau1", 0.95),
            tau2=getattr(args, "tau2", 0.80),
            w_q=getattr(args, "w_q", 1.0),
            w_r=getattr(args, "w_r", 0.3),
            w_c=w_c,
            min_rep=getattr(args, "min_rep", 0.0),
            max_k=max_k,
            max_retries=getattr(args, "max_retries", 1),
            eta_rep=getattr(args, "eta", 0.3),
            enable_short_code_prompt=not getattr(args, "no_short_code_prompt", False),
            enable_light_tests=not getattr(args, "no_light_tests", False),
            disable_cost_ranking=no_cost_aware,
            disable_reputation=no_rep,
            disable_diversity=getattr(args, "no_diversity", False),
            stage2_strategy=getattr(args, "stage2_strategy", "vote"),
            fallback_candidate_name=getattr(args, "fallback_candidate_name", ""),
            usal_mode=getattr(args, "usal_mode", "default"),
        )
    
    # Self-Consistency系列
    elif method.startswith("SC-"):
        k = int(method.split("-")[1])
        return await run_self_consistency(task, candidates, k)  # ✅ 直接调用自己的函数
    
    # Self-Ask
    elif method == "SA":
        # SA在baselines.py里是baseline_SA，需要通过exec.py调用
        # 这里创建一个wrapper
        from .baselines import baseline_SA
        agent = candidates[0]
        result = await baseline_SA(agent, task, [])
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": result.latency_s,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "agent_used": agent.model
        }
    
    # FrugalGPT (修复后)
    elif method == "FrugalGPT":
        return await run_frugal_gpt(task, candidates, args)
    
    # ❌ 删除RouteLLM
    # elif method == "RouteLLM":
    #     return await run_routellm(task, candidates, args)
    
    # 🆕 新增的Train-Free Baselines
    elif method == "Oracle":
        return await run_oracle(task, candidates, args)
    
    elif method == "Random":
        return await run_random_routing(task, candidates, args)
    
    elif method == "Cascade":
        return await run_cascade(task, candidates, args)

    elif method == "CodeExecRerank":
        return await run_code_exec_rerank(task, candidates, args)

    elif method == "CodeSelfRepair":
        return await run_code_self_repair(task, candidates, args)

    elif method == "AutoMixStyle":
        return await run_automix_style(task, candidates, args)

    elif method == "Smoothie":
        return await run_smoothie(task, candidates, args)

    elif method == "CP-Router":
        return await run_cp_router(task, candidates, args)
    
    elif method.startswith("MV-"):  # Majority Voting
        k = int(method.split("-")[1])
        return await run_majority_voting(task, candidates, k, args)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================
# TrustRoute 及其消融版本
# ============================================

async def run_trustroute(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """完整的 TrustRoute"""
    start_time = time.time()
    total_cost = 0.0
    
    # 使用完整的排名机制
    ranked = rank_agents_by_rep_cost(
        candidates,
        w_q=getattr(args, 'w_q', 1.0),
        w_r=getattr(args, 'w_r', 0.3),
        w_c=getattr(args, 'w_c', 0.2),
        min_rep=getattr(args, 'min_rep', 0.0)
    )
    
    if not ranked:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": "No qualified agents"
        }
    
    agent = ranked[0]
    prompt = task.get("prompt", "")
    
    try:
        result: AgentResult = await agent.generate(prompt)
        update_reputation(agent.model, success=True)
        
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time,
            "agent_used": agent.model
        }
    except Exception as e:
        update_reputation(agent.model, success=False)
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


async def run_trustroute_norep(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """TrustRoute 无信誉机制（消融实验）"""
    start_time = time.time()
    
    # 只考虑质量和成本，不考虑信誉
    scored = []
    for agent in candidates:
        cost_norm = agent.cpk / 0.1
        quality = agent.extra.get("quality", 0.7)
        score = 1.0 * quality - 0.2 * cost_norm  # 固定权重
        scored.append((score, agent))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    agent = scored[0][1]
    
    try:
        result: AgentResult = await agent.generate(task.get("prompt", ""))
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


async def run_trustroute_nocost(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """TrustRoute 无成本排序（消融实验）"""
    start_time = time.time()
    
    # 只考虑质量和信誉，忽略成本
    scored = []
    for agent in candidates:
        rep = get_agent_reputation(agent.model)
        quality = agent.extra.get("quality", 0.7)
        score = 1.0 * quality + 0.3 * rep  # 忽略成本项
        scored.append((score, agent))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    agent = scored[0][1]
    
    try:
        result: AgentResult = await agent.generate(task.get("prompt", ""))
        update_reputation(agent.model, success=True)
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time
        }
    except Exception as e:
        update_reputation(agent.model, success=False)
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


# ============================================
# 基础 Baseline 方法
# ============================================

async def run_single_agent(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """Single-Agent baseline（使用第一个/最好的模型）"""
    start_time = time.time()
    
    # 使用候选列表中的第一个模型
    agent = candidates[0]
    
    try:
        result: AgentResult = await agent.generate(task.get("prompt", ""))
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time,
            "agent_used": agent.model
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


async def run_self_consistency(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    k: int
) -> Dict[str, Any]:
    """Self-Consistency baseline (vote on extracted final answer key, not full text)."""
    import re
    from collections import Counter

    start_time = time.time()
    agent = candidates[0]
    ds = (task.get("dataset") or "").lower()

    # 1) prompt：GSM8K 强制 ####
    prompt = task.get("prompt", "") or task.get("question", "") or ""
    if ds == "gsm8k":
        prompt = prompt.rstrip() + "\n\nPlease solve step-by-step and output the final answer as: #### <number>"

    # 2) vote key：提取最后的“数字/分数”，用于投票
    numpat = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?")
    def vote_key(text: str) -> str:
        if not text:
            return "NO_ANSWER"
        tail = text.split("####")[-1] if "####" in text else text[-250:]
        nums = numpat.findall(tail)
        if not nums:
            return "NO_ANSWER"
        key = nums[-1].replace(",", "")
        key = re.sub(r"\s+", "", key)  # 统一 1 / 2 -> 1/2
        return key.strip()

    answers = []
    keys = []
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # 3) 逐次采样（保持你现在的 latency 口径：串行求和）
    for _ in range(k):
        try:
            result: AgentResult = await agent.generate(prompt, temperature=0.7)
            txt = (result.text or "").strip()
            if not txt:
                continue
            answers.append(txt)
            keys.append(vote_key(txt))

            total_cost += float(getattr(result, "cost_usd", 0.0) or 0.0)
            total_prompt_tokens += int(getattr(result, "prompt_tokens", 0) or 0)
            total_completion_tokens += int(getattr(result, "completion_tokens", 0) or 0)
        except Exception:
            continue

    if not answers:
        return {
            "candidate": "",
            "cost_usd": total_cost,
            "latency_s": time.time() - start_time,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "agent_used": agent.model,
            "note": f"SC-{k}-all_failed"
        }

    # 4) 多数投票（按 key）
    cnt = Counter(keys)
    best_key, best_votes = cnt.most_common(1)[0]
    # 多个同票，取最后一个（通常更像“最终答案”）
    idxs = [i for i, kk in enumerate(keys) if kk == best_key]
    chosen_idx = idxs[-1]
    final_answer = answers[chosen_idx]

    return {
        "candidate": final_answer,
        "cost_usd": total_cost,
        "latency_s": time.time() - start_time,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "agent_used": agent.model,
        "note": f"SC-{k} vote={best_votes}/{len(keys)} key={best_key}"
    }



# ============================================
# 高级 Baseline 方法（可选实现）
# ============================================

async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    import time
    start = time.time()
    
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    total_cost = 0
    best_result = None
    best_score = -1
    
    # ✅ 添加数据集特定的 prompt 处理
    base_prompt = task.get("prompt", "")
    dataset = task.get("dataset", "")
    
    for agent in sorted_agents:
        # 根据数据集调整 prompt
        prompt = base_prompt
        if dataset == "gsm8k":
            prompt += "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
        
        result = await agent.generate(prompt)  # ✅ 使用增强的 prompt


def compute_quality_score(text: str, dataset: str) -> float:
    """
    启发式质量评分（0-1之间）
    基于多个指标的加权组合
    """
    score = 0.0
    
    if dataset in ["mbpp", "humaneval"]:
        # 代码任务的质量指标
        has_def = "def " in text
        has_return = "return" in text or "yield" in text
        has_docstring = '"""' in text or "'''" in text
        reasonable_length = 50 < len(text) < 2000
        no_syntax_error = check_basic_syntax(text)
        
        score = (
            0.3 * has_def +
            0.3 * has_return +
            0.1 * has_docstring +
            0.2 * reasonable_length +
            0.1 * no_syntax_error
        )
    
    elif dataset == "gsm8k":
        # 数学任务的质量指标（修复：降低对格式的依赖）
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        has_final_number = bool(numbers)  # 只要有数字就行
        has_calculation = any(op in text for op in ['+', '-', '*', '/', '=', 'x'])
        has_steps = text.count('\n') >= 2
        reasonable_length = 50 < len(text) < 1000
        
        # ✅ 新评分标准（降低对 #### 的依赖）
        score = (
            0.3 * has_final_number +    # 30%: 有最终数字
            0.3 * has_calculation +      # 30%: 有计算过程
            0.2 * has_steps +            # 20%: 有步骤
            0.2 * reasonable_length      # 20%: 长度合理
        )
    
    return min(score, 1.0)


def check_basic_syntax(code: str) -> bool:
    """检查代码是否有明显语法错误"""
    try:
        import ast
        ast.parse(code)
        return True
    except:
        return False


async def run_routellm(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """RouteLLM baseline（简化实现）"""
    # TODO: 如果要实现完整的RouteLLM，需要训练胜率预测器
    # 这里提供简化版本：基于查询长度的启发式路由
    start_time = time.time()
    
    prompt = task.get("prompt", "")
    prompt_length = len(prompt.split())
    
    # 简单规则：短查询用便宜模型，长查询用贵模型
    if prompt_length < 50:
        agent = min(candidates, key=lambda a: a.cpk)
    else:
        agent = max(candidates, key=lambda a: a.extra.get("quality", 0.5))
    
    try:
        result: AgentResult = await agent.generate(prompt)
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time,
            "agent_used": agent.model
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }
async def run_oracle(task: dict, candidates: list, args) -> dict:
    """
    Oracle: 总是选择最强的模型（理论上界）
    
    假设：成本最高的模型 = 能力最强的模型
    """
    import time
    start = time.time()
    
    # ✅ 基于成本选择最强模型
    def get_model_cost(agent):
        """计算模型的平均成本"""
        # 尝试从不同的属性获取成本信息
        if hasattr(agent, 'cpk') and agent.cpk > 0:
            return agent.cpk
        
        # 尝试从 pricing 元数据
        if hasattr(agent, 'extra') and isinstance(agent.extra, dict):
            pricing = agent.extra.get('pricing', {})
            if pricing:
                cost_in = pricing.get('input', 0)
                cost_out = pricing.get('output', 0)
                return (cost_in + cost_out) / 2
        
        # 尝试直接获取价格属性
        cost_in = getattr(agent, 'prompt_price_per_1k', 0)
        cost_out = getattr(agent, 'completion_price_per_1k', 0)
        
        return (cost_in + cost_out) / 2
    
    # 选择成本最高的模型
    best_agent = max(candidates, key=get_model_cost)
    
    # 打印调试信息
    model_costs = [(getattr(a, 'model', str(a)), get_model_cost(a)) for a in candidates]
    print(f"[Oracle] Model costs: {model_costs}")
    print(f"[Oracle] Selected: {getattr(best_agent, 'model', str(best_agent))} "
          f"(cost: ${get_model_cost(best_agent):.6f}/1k)")
    
    # 只调用这一个模型
    result = await best_agent.generate(task.get("prompt", ""))
    
    return {
        "candidate": result.text,
        "cost_usd": result.cost_usd,
        "latency_s": time.time() - start,
        "agent_used": getattr(best_agent, 'model', str(best_agent)),
        "note": "Oracle"
    }


async def run_random_routing(task: dict, candidates: list, args) -> dict:
    """随机选择一个模型（理论下界）"""
    import random, time
    start = time.time()
    
    agent = random.choice(candidates)
    
    # ✅ 添加异常处理
    try:
        result = await agent.generate(task.get("prompt", ""))
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start,
            "agent_used": agent.model,
            "note": "Random"
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": str(e)
        }


async def run_cascade(task: dict, candidates: list, args) -> dict:
    """
    级联路由：从便宜到贵依次尝试，直到答案看起来valid
    """
    import time
    start = time.time()
    
    # 按成本排序
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    total_cost = 0
    last_result = None
    
    for agent in sorted_agents:
        try:
            result = await agent.generate(task.get("prompt", ""))
            total_cost += result.cost_usd
            last_result = result
            
            # ✅ 简单的valid检查
            if is_valid_answer(result.text, task.get("dataset")):
                return {
                    "candidate": result.text,
                    "cost_usd": total_cost,
                    "latency_s": time.time() - start,
                    "agent_used": agent.model,
                    "note": f"Cascade-Success-at-{agent.model}"
                }
        except Exception as e:
            continue
    
    # ✅ 如果都不valid或都失败，返回最后一个
    if last_result:
        return {
            "candidate": last_result.text,
            "cost_usd": total_cost,
            "latency_s": time.time() - start,
            "agent_used": sorted_agents[-1].model if sorted_agents else "unknown",
            "note": "Cascade-Fallback"
        }
    else:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": "All agents failed"
        }


async def run_majority_voting(task: dict, candidates: list, k: int, args) -> dict:
    """
    多数投票：并行调用k个模型，选择最高票的答案
    """
    import time, random
    from collections import Counter
    start = time.time()
    
    # 随机选择k个不同的模型
    selected = random.sample(candidates, min(k, len(candidates)))
    
    # ✅ 修复：添加异常处理
    results = await asyncio.gather(*[
        agent.generate(task.get("prompt", ""))
        for agent in selected
    ], return_exceptions=True)
    
    # ✅ 过滤异常结果
    valid_results = [r for r in results if not isinstance(r, Exception)]
    
    if not valid_results:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": "All agents failed"
        }
    
    # 提取答案并投票
    answers = [extract_final_answer(r.text, task.get("dataset")) for r in valid_results]
    answer_counts = Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]
    
    # 找到第一个给出这个答案的结果
    chosen_result = None
    for r, ans in zip(valid_results, answers):
        if ans == most_common_answer:
            chosen_result = r
            break
    
    # ✅ 防御性检查
    if chosen_result is None:
        chosen_result = valid_results[0]
    
    total_cost = sum(r.cost_usd for r in valid_results)
    
    return {
        "candidate": chosen_result.text,
        "cost_usd": total_cost,
        "latency_s": time.time() - start,
        "agent_used": f"MajorityVoting-k{k}",
        "note": f"Votes:{count}/{len(valid_results)}"
    }


# 辅助函数
def is_valid_answer(text: str, dataset: str) -> bool:
    """检查答案是否看起来valid（不需要ground truth）"""
    if dataset in ["mbpp", "humaneval"]:
        # 代码任务：必须包含函数定义
        return "def " in text and ("return" in text or "yield" in text)
    elif dataset == "gsm8k":
        # 数学任务：必须包含最终答案标记
        return "####" in text or any(char.isdigit() for char in text[-50:])
    return len(text.strip()) > 20


def extract_final_answer(text: str, dataset: str) -> str:
    """提取最终答案用于投票"""
    if dataset == "gsm8k":
        if "####" in text:
            return text.split("####")[-1].strip()
    elif dataset in ["mbpp", "humaneval"]:
        # 提取函数签名作为答案标识
        import re
        match = re.search(r'def\s+(\w+)\s*\(', text)
        if match:
            return match.group(1)
    return text[:100]  # 默认取前100字符


def load_ground_truth(task: dict) -> str:
    """加载ground truth（仅Oracle使用）"""
    dataset = task.get("dataset")
    if dataset == "gsm8k":
        # GSM8K的答案在原始数据中
        return task.get("answer", "")
    elif dataset in ["mbpp", "humaneval"]:
        # 代码任务的测试用例
        return task.get("test", "")
    return ""


def check_correctness(answer: str, ground_truth: str, dataset: str) -> bool:
    """检查答案是否正确（仅Oracle使用）"""
    if dataset == "gsm8k":
        # 提取数字答案
        import re
        pred_nums = re.findall(r'-?\d+\.?\d*', answer.split("####")[-1] if "####" in answer else answer)
        gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth)
        if pred_nums and gt_nums:
            return abs(float(pred_nums[-1]) - float(gt_nums[-1])) < 0.01
    elif dataset in ["mbpp", "humaneval"]:
        # 代码任务需要运行测试（这里简化为包含关键代码结构）
        return "def " in answer and "return" in answer
    return False

# ================= REPAIRED FUNCTIONS INJECTED BY FIX SCRIPT =================

async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    """
    FrugalGPT (Robust Version): 
    按成本排序，依次尝试。如果模型失败或超时，立即切换下一个。
    """
    import time
    import asyncio
    start = time.time()
    
    # 按成本排序 (便宜 -> 贵)
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    # 数据集特定的 Prompt 后缀
    base_prompt = task.get("prompt", "")
    dataset = task.get("dataset", "")
    suffix = ""
    if dataset == "gsm8k":
        suffix = "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
    
    full_prompt = base_prompt + suffix
    
    last_error = None
    
    for agent in sorted_agents:
        try:
            # 尝试生成
            # print(f"  [FrugalGPT] Trying {agent.model}...")
            result = await agent.generate(full_prompt)
            
            # 检查结果是否为空
            if not result.text or not result.text.strip():
                raise ValueError("Empty response from agent")
                
            # 简单验证 (Cascade 逻辑的一部分，FrugalGPT 也可以用)
            # 如果看起来是合法的（比如有代码或数字），就直接接受
            # 这里为了省钱，只要不报错且不为空，我们就接受 (或者你可以加 is_valid_answer 判断)
            
            if is_valid_answer(result.text, dataset):
                return {
                    "candidate": result.text,
                    "cost_usd": result.cost_usd,
                    "latency_s": time.time() - start,
                    "agent_used": agent.model,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "note": "Frugal-Success"
                }
            else:
                # 结果格式不对，视为失败，尝试下一个昂贵模型
                # print(f"  [FrugalGPT] {agent.model} output invalid, escalating...")
                continue
                
        except Exception as e:
            last_error = e
            # print(f"  [FrugalGPT] {agent.model} failed: {e}, trying next...")
            continue
            
    # 如果所有模型都失败了，返回最后一个异常
    return {
        "candidate": "",
        "cost_usd": 0,
        "latency_s": time.time() - start,
        "error": f"All agents failed. Last error: {str(last_error)}"
    }


async def run_cascade(task: dict, candidates: list, args) -> dict:
    """
    Cascade (Robust Version):
    依次尝试，直到 is_valid_answer 为真。遇到超时直接跳过。
    """
    import time
    start = time.time()
    
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    total_cost = 0
    last_result = None
    last_error = None
    
    dataset = task.get("dataset", "")
    prompt = task.get("prompt", "")
    
    for agent in sorted_agents:
        try:
            result = await agent.generate(prompt)
            total_cost += result.cost_usd
            
            # 检查是否为空
            if not result.text or not result.text.strip():
                continue
                
            last_result = result
            
            # 有效性检查
            if is_valid_answer(result.text, dataset):
                return {
                    "candidate": result.text,
                    "cost_usd": total_cost,
                    "latency_s": time.time() - start,
                    "agent_used": agent.model,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "note": f"Cascade-Success-at-{agent.model}"
                }
        except Exception as e:
            last_error = e
            continue
    
    # 如果都失败了，但在过程中有产生过（无效）结果，返回最后一个
    if last_result:
        return {
            "candidate": last_result.text,
            "cost_usd": total_cost,
            "latency_s": time.time() - start,
            "agent_used": sorted_agents[-1].model if sorted_agents else "unknown",
            "prompt_tokens": last_result.prompt_tokens,
            "completion_tokens": last_result.completion_tokens,
            "note": "Cascade-Fallback-Invalid"
        }
    else:
        return {
            "candidate": "",
            "cost_usd": total_cost,
            "latency_s": time.time() - start,
            "error": f"All agents failed. Last error: {str(last_error)}"
        }




# [PATCHED BY AUTO-SCRIPT]
async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    """
    FrugalGPT (Robust Fixed): 
    按成本排序。如果模型超时/空结果，立即尝试下一个，绝不崩溃。
    """
    import time
    import asyncio
    start = time.time()
    
    # 1. 按成本排序
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    # 2. 准备 Prompt (GSM8K特化)
    base_prompt = task.get("prompt", "")
    dataset = task.get("dataset", "")
    suffix = ""
    if dataset == "gsm8k":
        suffix = "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
    full_prompt = base_prompt + suffix
    
    last_error = None
    
    for agent in sorted_agents:
        try:
            # print(f"  [Frugal] Trying {agent.model}...")
            # 3. 调用模型
            result = await agent.generate(full_prompt)
            
            # 4. 检查是否为空 (关键修复点!)
            if not result.text or not result.text.strip():
                # print(f"  [Frugal] {agent.model} returned empty, skipping...")
                continue
            
            # 5. 只要有内容，就视为成功 (交给后续评价指标去判断对错)
            return {
                "candidate": result.text,
                "cost_usd": result.cost_usd,
                "latency_s": time.time() - start,
                "agent_used": agent.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "note": "Frugal-Success"
            }

        except Exception as e:
            last_error = e
            # print(f"  [Frugal] {agent.model} error: {e}, skipping...")
            continue
            
    # 6. 全军覆没兜底
    return {
        "candidate": "",
        "cost_usd": 0,
        "latency_s": time.time() - start,
        "error": f"All agents failed. Last error: {str(last_error)}"
    }




# [INJECTED ROBUST FIX]
async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    """FrugalGPT (Robust): Auto-retry on failure, never crash."""
    import time, asyncio
    start = time.time()
    
    # 按成本排序
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    # GSM8K Prompt patch
    suffix = ""
    if task.get("dataset") == "gsm8k":
        suffix = "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
    full_prompt = (task.get("prompt", "") or "") + suffix
    
    last_error = None
    
    for agent in sorted_agents:
        try:
            # print(f"Trying {agent.model}...")
            result = await agent.generate(full_prompt)
            
            # 关键：检查是否为空
            if not result.text or not result.text.strip():
                continue # 空结果，跳过，找下一个模型
            
            return {
                "candidate": result.text,
                "cost_usd": result.cost_usd,
                "latency_s": time.time() - start,
                "agent_used": agent.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "note": "Frugal-Success"
            }
        except Exception as e:
            last_error = e
            # 出错不崩溃，继续循环找下一个模型
            continue
            
    # 全失败时的兜底
    return {
        "candidate": "",
        "cost_usd": 0,
        "latency_s": time.time() - start,
        "error": f"All agents failed. Last: {str(last_error)}"
    }
