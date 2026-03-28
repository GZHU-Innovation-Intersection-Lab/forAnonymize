import os
import json
import re
from typing import Dict, Optional, List, Any, Tuple  # ✅ 添加 List, Any, Tuple

def get_rep_file_path() -> str:
    """获取信誉文件路径（基于环境变量 ISSTA_SEED）"""
    seed = os.environ.get("ISSTA_SEED", "1")
    rep_tag = os.environ.get("ISSTA_REP_TAG", "").strip()
    rep_tag_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", rep_tag)[:80]
    cache_dir = os.path.join(os.getcwd(), "ISSTA", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    if rep_tag_safe:
        return os.path.join(cache_dir, f"rep_state_{rep_tag_safe}_seed{seed}.json")
    return os.path.join(cache_dir, f"rep_state_seed{seed}.json")

def load_rep_state() -> Dict[str, Dict]:
    """加载信誉状态"""
    path = get_rep_file_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_rep_state(state: Dict[str, Dict]):
    """保存信誉状态"""
    path = get_rep_file_path()
    with open(path, "w") as f:
        json.dump(state, f, indent=2)

def reset_rep_state():
    """重置信誉状态"""
    path = get_rep_file_path()
    if os.path.exists(path):
        os.remove(path)
        print(f"[TrustRoute] 🗑️  Deleted existing: {path}")
    else:
        print(f"[TrustRoute] ℹ️  No existing reputation state to reset")

def get_agent_reputation(model_id: str) -> float:
    """获取代理信誉分数"""
    state = load_rep_state()
    if model_id not in state:
        return 0.5  # 默认初始信誉
    
    data = state[model_id]
    success = data.get("success", 0)
    total = data.get("total", 0)
    
    if total == 0:
        return 0.5
    
    return success / total

def update_reputation(model_id: str, success: bool):
    """更新代理信誉"""
    state = load_rep_state()
    
    if model_id not in state:
        state[model_id] = {"success": 0, "total": 0}
    
    state[model_id]["total"] += 1
    if success:
        state[model_id]["success"] += 1
    
    save_rep_state(state)
def get_rep(state: Dict[str, Dict], agent_name: str) -> float:
    """
    获取 USAL 信誉分数（0-100）
    
    兼容旧格式：如果没有 usal_score，从 success/total 计算
    """
    if agent_name not in state:
        return 50.0  # 默认中等信誉
    
    data = state[agent_name]
    
    # 优先使用 USAL 分数
    if "usal_score" in data:
        return data["usal_score"]
    
    # 降级：从 success/total 计算（归一化到 0-100）
    success = data.get("success", 0)
    total = data.get("total", 0)
    
    if total == 0:
        return 50.0
    
    # 转换为 0-100 分数
    return (success / total) * 100.0


def set_rep(state: Dict[str, Dict], agent_name: str, score: float):
    """
    设置 USAL 信誉分数（0-100）
    """
    if agent_name not in state:
        state[agent_name] = {"success": 0, "total": 0}
    
    # 限制范围
    state[agent_name]["usal_score"] = max(0.0, min(100.0, score))


def rank_agents_by_rep_cost(
    agents: List[Dict[str, Any]],
    rep_state: Dict[str, Dict],
    w_q: float = 1.0,
    w_r: float = 0.3,
    w_c: float = 0.2,
    min_rep: float = 0.0
) -> List[Dict[str, Any]]:
    """
    按信誉和成本对 agents 排序
    
    输入格式：
        agents: [{"name": "gpt-4", "model": "gpt-4", "obj": BaseAgent, ...}, ...]
    
    排序策略：
    1. 信誉高的优先
    2. 信誉相近时（差异<5），成本低的优先
    """
    mr = float(min_rep)
    if mr > 1.0:
        mr = mr / 100.0

    scored: List[Tuple[float, Dict[str, Any], float, float, float]] = []
    for agent in agents:
        name = agent.get("name", "")
        rep_01 = get_rep(rep_state, name) / 100.0
        if rep_01 < mr:
            continue

        pricing = agent.get("meta", {}).get("pricing", {})
        cost_in = float(pricing.get("input", 0.00015))
        cost_out = float(pricing.get("output", 0.00060))
        avg_cost = (cost_in + cost_out) / 2.0
        cost_norm = avg_cost / 0.001

        agent_obj = agent.get("obj")
        extra = getattr(agent_obj, "extra", {}) if agent_obj is not None else {}
        quality = float(extra.get("quality", 0.7)) if isinstance(extra, dict) else 0.7

        score = (w_q * quality) + (w_r * rep_01) - (w_c * cost_norm)
        scored.append((score, agent, rep_01, avg_cost, quality))

    scored.sort(key=lambda x: x[0], reverse=True)
    sorted_agents = [a for _, a, _, _, _ in scored]
    
    print(f"[Agent Ranking] Top 3:")
    for i, agent in enumerate(sorted_agents[:3]):
        name = agent.get("name", "")
        rep = get_rep(rep_state, name)
        pricing = agent.get("meta", {}).get("pricing", {})
        cost = (pricing.get("input", 0) + pricing.get("output", 0)) / 2
        print(f"  {i+1}. {name}: rep={rep:.1f}, cost=${cost:.5f}")
    
    return sorted_agents
