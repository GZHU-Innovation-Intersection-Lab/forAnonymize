# src/utils/telemetry.py
from dataclasses import dataclass

@dataclass
class CallStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0

def priced_cost(pt: int, ct: int, price_in: float, price_out: float) -> float:
    return (pt*price_in + ct*price_out) / 1_000_000.0

def priced_call(llm_fn, prompt: str, price_in: float, price_out: float):
    """
    llm_fn: callable(prompt) -> dict(text, prompt_tokens, completion_tokens, latency_s)
    返回 (text, stats)
    """
    out = llm_fn(prompt)
    pt = int(out.get("prompt_tokens", 0))
    ct = int(out.get("completion_tokens", 0))
    lat = float(out.get("latency_s", 0.0))
    cost = priced_cost(pt, ct, price_in, price_out)
    stats = CallStats(prompt_tokens=pt, completion_tokens=ct, latency_s=lat, cost_usd=cost)
    return out.get("text", ""), stats