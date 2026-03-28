# src/tasks/loader.py
from typing import List, Dict, Any, Optional
import random

def _sample_gsm8k() -> List[Dict[str, Any]]:
    # 极简版 GSM8K 样例（可自行扩充）
    data = [
        {
            "id": "gsm8k-1",
            "name": "gsm8k-1",
            "dataset": "gsm8k",
            "question": "John had 3 apples and bought 4 more. How many apples does he have now?",
            "prompt": "Q: John had 3 apples and bought 4 more. How many apples does he have now?\nA:",
            "input": "John had 3 apples and bought 4 more. How many apples now?",
            "reference": "7",
        },
        {
            "id": "gsm8k-2",
            "name": "gsm8k-2",
            "dataset": "gsm8k",
            "question": "A box has 12 candies. Tom eats 5. How many candies remain?",
            "prompt": "Q: A box has 12 candies. Tom eats 5. How many candies remain?\nA:",
            "input": "A box has 12 candies. Tom eats 5. How many remain?",
            "reference": "7",
        },
        {
            "id": "gsm8k-3",
            "name": "gsm8k-3",
            "dataset": "gsm8k",
            "question": "If there are 8 birds on a tree and 3 fly away, how many are left?",
            "prompt": "Q: If there are 8 birds on a tree and 3 fly away, how many are left?\nA:",
            "input": "8 birds on a tree; 3 fly away. How many left?",
            "reference": "5",
        },
    ]
    return data

def _sample_humaneval() -> List[Dict[str, Any]]:
    # 极简版 HumanEval 风格：生成一个简短函数
    data = [
        {
            "id": "humaneval-1",
            "name": "humaneval-1",
            "dataset": "humaneval",
            "prompt": (
                "Write a Python function add(a, b) that returns the sum of a and b.\n"
                "Provide only the function definition."
            ),
            "question": "Implement function add(a, b).",
            "input": "",
            "reference": "def add(a, b):\n    return a + b\n",
        },
        {
            "id": "humaneval-2",
            "name": "humaneval-2",
            "dataset": "humaneval",
            "prompt": (
                "Write a Python function is_even(n) that returns True if n is even, else False.\n"
                "Provide only the function definition."
            ),
            "question": "Implement function is_even(n).",
            "input": "",
            "reference": "def is_even(n):\n    return n % 2 == 0\n",
        },
    ]
    return data

def _sample_mbpp() -> List[Dict[str, Any]]:
    # 极简版 MBPP 风格：简单算法题
    data = [
        {
            "id": "mbpp-1",
            "name": "mbpp-1",
            "dataset": "mbpp",
            "prompt": (
                "Write a Python function factorial(n) that returns n! (assume n is a non-negative integer).\n"
                "Provide only the function definition."
            ),
            "question": "Implement factorial(n).",
            "input": "",
            "reference": (
                "def factorial(n):\n"
                "    res = 1\n"
                "    for i in range(2, n+1):\n"
                "        res *= i\n"
                "    return res\n"
            ),
        },
        {
            "id": "mbpp-2",
            "name": "mbpp-2",
            "dataset": "mbpp",
            "prompt": (
                "Write a Python function reverse_string(s) that returns the reversed string.\n"
                "Provide only the function definition."
            ),
            "question": "Implement reverse_string(s).",
            "input": "",
            "reference": "def reverse_string(s):\n    return s[::-1]\n",
        },
    ]
    return data

def _normalize_names(names: List[str]) -> List[str]:
    norm = []
    for n in names:
        x = (n or "").strip().lower()
        if x in {"gsm8k"}:
            norm.append("gsm8k")
        elif x in {"humaneval", "human-eval", "human_eval"}:
            norm.append("humaneval")
        elif x in {"mbpp"}:
            norm.append("mbpp")
        else:
            # 未知数据集名直接原样保留，后面会给出空列表
            norm.append(x)
    return norm

def load_tasks(datasets: List[str], limit: int = 0, seeds: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    最小可用的任务加载器：
    - 支持数据集：gsm8k / humaneval / mbpp
    - 如需跑更多样本，请自行扩充上面的 _sample_* 列表或改为读取文件。
    - 输出：{dataset_name: [task, ...]}，task 至少包含 id/name/prompt 字段，其它字段可选。
    """
    seed_value = None
    if seeds:
        try:
            seed_value = int(seeds[0])
        except Exception:
            seed_value = None

    rng = random.Random(seed_value)
    out: Dict[str, List[Dict[str, Any]]] = {}

    for ds in _normalize_names(datasets):
        if ds == "gsm8k":
            items = _sample_gsm8k()
        elif ds == "humaneval":
            items = _sample_humaneval()
        elif ds == "mbpp":
            items = _sample_mbpp()
        else:
            items = []

        # 稳定随机打乱（按 seed）
        rng.shuffle(items)

        # 应用 limit
        if limit and limit > 0:
            items = items[:limit]

        out[ds] = items

    return out