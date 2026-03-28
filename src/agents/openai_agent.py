# /data_huawei/jiakun/AAMAS/src/agents/openai_agent.py
import os, time, asyncio
from typing import Optional
from openai import OpenAI
from .base import BaseAgent, AgentResult
from ..utils.tokens import approx_tokens, cost_usd

class OpenAIChatAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def generate(self, prompt: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> AgentResult:
        t0 = time.time()
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        messages = []
        if self.system_prompt:
            messages.append({"role":"system","content": self.system_prompt})
        messages.append({"role":"user","content": prompt})
        resp = await asyncio.to_thread(self.client.chat.completions.create,
            model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        choice = resp.choices[0]
        text = choice.message.content or ""
        pt = resp.usage.prompt_tokens if hasattr(resp, "usage") else approx_tokens(str(messages))
        ct = resp.usage.completion_tokens if hasattr(resp, "usage") else approx_tokens(text)
        return AgentResult(
            text=text, prompt_tokens=pt, completion_tokens=ct,
            cost_usd=cost_usd(pt, ct, self.ppk, self.cpk),
            latency_s=time.time()-t0, finish_reason=choice.finish_reason or "stop"
        )