#/data_huawei/jiakun/AAMAS/src/fusion/fusion.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Candidate:
    agent_name: str
    text: str
    cost_usd: float
    latency_s: float
    quality: float
    safety: float
    rep: float

def minmax_norm(x: float, lo: float, hi: float) -> float:
    if hi<=lo: return 0.0
    return (x-lo)/(hi-lo)

def select_with_safety(cands: List[Candidate], weights: Dict[str, float], tau: float) -> Candidate | None:
    # Filter by safety threshold
    pool = [c for c in cands if c.safety >= tau]
    if not pool:
        # fallback to max safety candidate (label high risk at caller)
        pool = sorted(cands, key=lambda c: c.safety, reverse=True)[:1]
    # Normalize cost for fusion score
    costs = [c.cost_usd for c in pool]
    lo,hi = min(costs), max(costs)
    best, best_score = None, -1e9
    for c in pool:
        cost_norm = minmax_norm(c.cost_usd, lo, hi) if hi>0 else 0.0
        score = weights["q"]*c.quality + weights["r"]*c.rep - weights["c"]*cost_norm
        if score > best_score:
            best, best_score = c, score
    return best