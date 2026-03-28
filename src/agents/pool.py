# src/agents/pool.py
import yaml
from typing import List
from .base import BaseAgent, MockAgent

try:
    from .gptsapi_agent import GptsApiAgent
except Exception:
    GptsApiAgent = None

try:
    from .openai_agent import OpenAIChatAgent
except Exception:
    OpenAIChatAgent = None

def build_agents(config_path: str) -> List[BaseAgent]:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    agents = []
    personas = {p["name"]: p for p in cfg.get("personas", [])}
    for a in cfg["agents"]:
        provider = a.get("provider", "mock").lower()
        persona = a.get("persona", "neutral")
        system = personas.get(persona, {}).get("system", "")
        delta_t = personas.get(persona, {}).get("delta_temp", 0.0)
        role = a.get("role", "candidate")
        extra_keys = [
            "base_url",
            "api_key_env",
            "role",
            "quality",
            "tier",
            "persona",
        ]
        kwargs = dict(
            name=a["name"],
            model=a["model"],
            temperature=a.get("temperature", 0.2) + delta_t,
            max_tokens=a.get("max_tokens", 512),
            prompt_price_per_1k=a.get("prompt_price_per_1k", 0.0),
            completion_price_per_1k=a.get("completion_price_per_1k", 0.0),
            base_latency_ms=a.get("base_latency_ms", 300),
            system_prompt=system,
            extra={k: a[k] for k in extra_keys if k in a}
        )
        if provider == "gptsapi" and GptsApiAgent is not None:
            agent = GptsApiAgent(**kwargs)
        elif provider == "openai" and OpenAIChatAgent is not None:
            agent = OpenAIChatAgent(**kwargs)
        elif provider == "mock":
            agent = MockAgent(**kwargs)
        else:
            agent = MockAgent(**kwargs)
        agent.extra["role"] = role
        agents.append(agent)
    return agents
