# /data_huawei/jiakun/AAMAS/src/utils/tokens.py
from typing import Tuple
try:
    import tiktoken
except Exception:
    tiktoken = None

def approx_tokens(text: str) -> int:
    if tiktoken:
        enc = tiktoken.get_encoding("cl100k_base")
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # fallback: rough 4 chars per token
    return max(1, len(text) // 4)

def cost_usd(prompt_tokens: int, completion_tokens: int, ppk: float, cpk: float) -> float:
    return (prompt_tokens/1000.0)*ppk + (completion_tokens/1000.0)*cpk