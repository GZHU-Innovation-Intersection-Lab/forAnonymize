# /data_huawei/jiakun/AAMAS/src/agents/base.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time, asyncio
from ..utils.tokens import approx_tokens, cost_usd

@dataclass
class AgentResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_s: float
    finish_reason: str

class BaseAgent:
    def __init__(self, name: str, model: str, temperature: float, max_tokens: int,
                 prompt_price_per_1k: float, completion_price_per_1k: float, base_latency_ms: int = 300,
                 system_prompt: Optional[str] = None, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ppk = prompt_price_per_1k
        self.cpk = completion_price_per_1k
        self.base_latency_ms = base_latency_ms
        self.system_prompt = system_prompt or ""
        self.extra = extra or {}

    async def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> AgentResult:
        raise NotImplementedError

class MockAgent(BaseAgent):
    async def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> AgentResult:
        t0 = time.time()
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        # Simple echo with minor transformation to emulate variability
        content = f"[{self.name}/{self.model} temp={temperature}] " + prompt.strip()[:1000]
        completion = " Answer: " + ("42" if "GSM8K" in prompt or "Question" in prompt else "def solution():\n    return 0")
        text = content + completion
        pt = approx_tokens(prompt + self.system_prompt)
        ct = min(approx_tokens(completion), max_tokens)
        time.sleep(self.base_latency_ms/1000.0)
        return AgentResult(
            text=text, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=cost_usd(pt, ct, self.ppk, self.cpk),
            latency_s=time.time()-t0, finish_reason="stop"
        )