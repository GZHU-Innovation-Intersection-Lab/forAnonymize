# src/runner/baselines.py
import asyncio
import re
from typing import Dict, Any, List, Optional
from src.agents.base import BaseAgent, AgentResult

_GSM8K_ENFORCE_FINAL_TAG: bool = True

def set_gsm8k_final_tag(flag: bool) -> None:
    global _GSM8K_ENFORCE_FINAL_TAG
    _GSM8K_ENFORCE_FINAL_TAG = bool(flag)

def _task_to_prompt(task: Dict[str, Any]) -> str:
    if isinstance(task, dict):
        p = task.get("prompt") or task.get("question") or task.get("input") or ""
        return p.strip()
    return str(task)

def build_prompt(task: Dict[str, Any]) -> str:
    prompt = _task_to_prompt(task)
    ds = (task.get("dataset") or task.get("name") or "").strip().lower()
    tid = str(task.get("task_id") or "").lower()
    
    if "gsm" in ds or "gsm" in tid:
        suffix = (
            "\n\nInstruction:\n"
            "Let's think step by step to solve this math problem.\n"
            "1. Break down the reasoning.\n"
            "2. Perform the arithmetic carefully.\n"
            "3. At the end, output the answer exactly as: #### <number>\n"
        )
        return f"{prompt.rstrip()}\n{suffix}"
            
    return prompt

async def assess_quality(task, text, judges=None):
    return 5.0

async def baseline_SC(agent: BaseAgent, task: Dict[str, Any], judges: List[BaseAgent], n: int = 3) -> AgentResult:
    prompt = build_prompt(task)
    
    original_sys_prompt = getattr(agent, "system_prompt", "")
    
    ds = (task.get("dataset") or "").lower()
    tid = str(task.get("task_id") or "").lower()
    if "gsm" in ds or "gsm" in tid:
        agent.system_prompt = "You are a helpful math assistant. You solve problems step-by-step in plain text."
    
    try:
        # 并发生成 n 次
        coros = [agent.generate(prompt, temperature=0.7) for _ in range(n)]
        outs = await asyncio.gather(*coros, return_exceptions=True)
        
        valid = [o for o in outs if isinstance(o, AgentResult) and o.text]
        if not valid:
            return AgentResult(text="", cost_usd=0.0, latency_s=0.0, prompt_tokens=0, completion_tokens=0, finish_reason="stop")
        
        # 🔧 修复：累加所有成本和 tokens
        chosen = valid[0]
        total_cost = sum(o.cost_usd for o in outs if isinstance(o, AgentResult))
        total_latency = max((o.latency_s for o in outs if isinstance(o, AgentResult)), default=0.0)
        total_prompt_tokens = sum(o.prompt_tokens for o in outs if isinstance(o, AgentResult))  # ✅ 累加
        total_completion_tokens = sum(o.completion_tokens for o in outs if isinstance(o, AgentResult))  # ✅ 累加
        
        return AgentResult(
            text=chosen.text,
            cost_usd=total_cost,
            latency_s=total_latency,
            prompt_tokens=total_prompt_tokens,  # ✅ 修复
            completion_tokens=total_completion_tokens,  # ✅ 修复
            finish_reason="stop"
        )
    finally:
        agent.system_prompt = original_sys_prompt

async def baseline_SA(a, t, j): return await baseline_SC(a, t, j, n=1)
async def baseline_MV(a, t, j, n=3): return await baseline_SC(a, t, j, n=n)
async def baseline_GEVal(a, t, j, n=3): return await baseline_SC(a, t, j, n=n)