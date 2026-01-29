#/data_huawei/jiakun/ISSTA/src/runner/exec.py
import argparse
import asyncio
from typing import List, Dict, Any
from .baselines import baseline_SA, baseline_SC
from .ours_lite_v2 import run_ours_lite
from ..agents.base import BaseAgent

class SafetyStub:
    async def score(self, src, out): return {"safety": 1.0}

async def _wrap(res, task):
    return {
        "candidate": res.text,
        "cost_usd": res.cost_usd,
        "latency_s": res.latency_s,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens
    }

def _safe_get(obj, key, default):
    val = getattr(obj, key, None)
    if val is not None: return float(val)
    if hasattr(obj, "extra") and isinstance(obj.extra, dict):
        val = obj.extra.get(key)
        if val is not None: return float(val)
    return default

async def run_method(method: str,
                     task: Dict[str, Any],
                     agents: List[BaseAgent],
                     judges: Dict[str, Any],
                     detruss: Any,
                     budget_usd: float,
                     tau: float,
                     weights: Any,
                     args: argparse.Namespace) -> Dict[str, Any]:
    
    # 1. Baselines (SA 和 SC 已经有 try-finally，不需要改)
    if method == "SA":
        old_sys = getattr(agents[0], "system_prompt", "")
        if "gsm" in str(task.get("dataset","")).lower():
            agents[0].system_prompt = "You are a helpful math assistant. Solve step by step."
        try:
            res = await baseline_SA(agents[0], task, [])
            return await _wrap(res, task)
        finally:
            agents[0].system_prompt = old_sys

    if method.startswith("SC-"):
        try: n = int(method.split("-")[1])
        except: n = 3
        
        old_sys = getattr(agents[0], "system_prompt", "")
        if "gsm" in str(task.get("dataset","")).lower():
            agents[0].system_prompt = "You are a helpful math assistant. Solve step by step."
            
        try:
            res = await baseline_SC(agents[0], task, [], n=n)
            return await _wrap(res, task)
        finally:
            agents[0].system_prompt = old_sys
    
    # 2. TrustRoute - 🔧 修复：添加 try-finally
    if method in ["Ours-Full", "TrustRoute", "Ours-Lite", "TrustRoute-NoRep", "TrustRoute-NoCostAware", "TrustRoute-NoParallel"]:
        # 保存原始 system_prompt
        original_prompts = {}
        if "gsm" in str(task.get("dataset","")).lower():
            for agent in agents:
                if hasattr(agent, "system_prompt"):
                    original_prompts[agent.name] = agent.system_prompt
                    agent.system_prompt = "You are a helpful math assistant. Solve step by step."
        
        try:
            # 原有的 TrustRoute 逻辑
            no_rep = "NoRep" in method
            no_cost = "NoCostAware" in method
            max_k = 1 if "NoParallel" in method else args.max_k
            
            agent_dicts = []
            for a in agents:
                p_in = _safe_get(a, "prompt_price_per_1k", 0.00015)
                p_out = _safe_get(a, "completion_price_per_1k", 0.00060)
                tier = getattr(a, "extra", {}).get("tier", "")
                
                agent_dicts.append({
                    "name": a.name,
                    "model": a.model,
                    "obj": a,
                    "tier": tier,
                    "meta": {"pricing": {"input": p_in, "output": p_out}}
                })

            async def gen_fn(ad, t, suf, temp):
                p = (t.get("prompt") or t.get("question") or str(t)) + suf
                r = await ad["obj"].generate(p, temperature=temp)
                return {"text": r.text, "cost_usd": r.cost_usd}
                
            async def judge_fn(t, txt): return [{"score": 0.5}]
            
            return await run_ours_lite(
                task, agent_dicts, gen_fn, judge_fn,
                budget_usd=budget_usd, tau1=args.tau1, tau2=args.tau2,
                max_k=max_k, disable_reputation=no_rep,
                disable_cost_ranking=no_cost,
                enable_short_code_prompt=True,
                enable_light_tests=True
            )
        finally:
            # 🔧 恢复原始 system_prompt
            for agent in agents:
                if agent.name in original_prompts:
                    agent.system_prompt = original_prompts[agent.name]