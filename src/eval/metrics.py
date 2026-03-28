# /src/eval/metrics.py
import re, contextlib, io, time, sys, types
from typing import Dict, Any

def extract_gsm8k_final(ans: str) -> str:
    m = re.findall(r"-?\d+(?:\.\d+)?", ans.replace(",", ""))
    return m[-1] if m else ans.strip().lower()

def acc_gsm8k(pred: str, gold: str) -> float:
    return 1.0 if extract_gsm8k_final(pred)==str(gold) else 0.0

def safe_exec(code: str, test_code: str, timeout_s: float = 2.0) -> bool:
    # Extremely simple and not bulletproof; for controlled CI only
    # Create a restricted global namespace
    allowed_builtins = {"range": range, "len": len, "print": print, "abs": abs, "min": min, "max": max, "sum": sum, "int": int, "float": float, "str": str, "list": list, "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate}
    g = {"__builtins__": allowed_builtins}
    loc = {}
    import threading

    exc = {"err": None, "ok": False}

    def run():
        try:
            exec(code, g, loc)
            exec(test_code, g, loc)
            exc["ok"] = True
        except Exception as e:
            exc["err"] = e

    th = threading.Thread(target=run)
    th.start()
    th.join(timeout=timeout_s)
    if th.is_alive():
        return False
    return exc["ok"]

def pass_at_1(code: str, tests: str) -> float:
    return 1.0 if safe_exec(code, tests) else 0.0