# src/runner/executor.py - 包含所有实验方法

import random
import time
import asyncio
from typing import List, Dict, Any, Optional
from collections import Counter
from src.agents.base import BaseAgent, AgentResult
from src.runner.lite_utils import (
    load_rep_state,
    save_rep_state,
    update_reputation,
    get_agent_reputation
)

def rank_agents_by_rep_cost(
    agents: List[BaseAgent],
    w_q: float = 1.0,
    w_r: float = 0.3,
    w_c: float = 0.2,
    min_rep: float = 0.0
) -> List[BaseAgent]:
    """根据质量、信誉和成本对代理进行排名"""
    scored = []
    for agent in agents:
        model_id = agent.model
        rep = get_agent_reputation(model_id)
        
        if rep < min_rep:
            continue
        
        cost_norm = agent.cpk / 0.1
        quality = agent.extra.get("quality", 0.7)
        score = w_q * quality + w_r * rep - w_c * cost_norm
        
        scored.append((score, agent))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [agent for _, agent in scored]


async def run_method(method: str, task: dict, candidates: list,
                     rep_state: dict, router, budget_usd: float, 
                     tau: float, judge, args) -> dict:
    """统一的方法调度器"""
    
    # TrustRoute系列
    if "TrustRoute" in method or "Ours" in method:
        from .ours_lite_v2 import run_ours_lite
        
        # 将 BaseAgent 对象转换为字典格式
        agent_dicts = []
        for agent in candidates:
            def _safe_get(obj, key, default):
                val = getattr(obj, key, None)
                if val is not None:
                    return float(val)
                if hasattr(obj, "extra") and isinstance(obj.extra, dict):
                    val = obj.extra.get(key)
                    if val is not None:
                        return float(val)
                return default
            
            p_in = _safe_get(agent, "prompt_price_per_1k", 0.00015)
            p_out = _safe_get(agent, "completion_price_per_1k", 0.00060)
            tier = getattr(agent, "extra", {}).get("tier", "unknown")
            
            agent_dicts.append({
                "name": agent.name,
                "model": agent.model,
                "obj": agent,
                "tier": tier,
                "meta": {
                    "pricing": {
                        "input": p_in,
                        "output": p_out
                    }
                }
            })
        
        # 🔧 修复：参数名改为 temperature
        async def gen_fn(agent_dict, task, suffix, temperature):
            agent_obj = agent_dict["obj"]
            prompt = (task.get("prompt") or task.get("question") or str(task)) + suffix
            result = await agent_obj.generate(prompt, temperature=temperature)
            return {
                "text": result.text,
                "cost_usd": result.cost_usd,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens
            }
        
        async def judge_fn(task, text):
            return [{"score": 0.5}]
        
        no_rep = "NoRep" in method
        no_cost = "NoCostAware" in method or "NoCost" in method
        no_parallel = "NoParallel" in method
        
        max_k = 1 if no_parallel else getattr(args, "max_k", 3)
        
        return await run_ours_lite(
            task=task,
            candidate_agents=agent_dicts,
            generate_fn=gen_fn,
            judge_fn=judge_fn,
            budget_usd=getattr(args, "budget_usd", 5.0),
            tau1=getattr(args, "tau1", 0.95),
            tau2=getattr(args, "tau2", 0.80),
            max_k=max_k,
            max_retries=getattr(args, "max_retries", 1),
            eta_rep=getattr(args, "eta", 0.3),
            enable_short_code_prompt=not getattr(args, "no_short_code_prompt", False),
            enable_light_tests=not getattr(args, "no_light_tests", False),
            disable_cost_ranking=no_cost,
            disable_reputation=no_rep,
            disable_diversity=getattr(args, "no_diversity", False)
        )
    
    # Self-Consistency系列
    elif method.startswith("SC-"):
        k = int(method.split("-")[1])
        return await run_self_consistency(task, candidates, k)  # ✅ 直接调用自己的函数
    
    # Self-Ask
    elif method == "SA":
        # SA在baselines.py里是baseline_SA，需要通过exec.py调用
        # 这里创建一个wrapper
        from .baselines import baseline_SA
        agent = candidates[0]
        result = await baseline_SA(agent, task, [])
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": result.latency_s,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "agent_used": agent.model
        }
    
    # FrugalGPT (修复后)
    elif method == "FrugalGPT":
        return await run_frugal_gpt(task, candidates, args)
    
    # ❌ 删除RouteLLM
    # elif method == "RouteLLM":
    #     return await run_routellm(task, candidates, args)
    
    # 🆕 新增的Train-Free Baselines
    elif method == "Oracle":
        return await run_oracle(task, candidates, args)
    
    elif method == "Random":
        return await run_random_routing(task, candidates, args)
    
    elif method == "Cascade":
        return await run_cascade(task, candidates, args)
    
    elif method.startswith("MV-"):  # Majority Voting
        k = int(method.split("-")[1])
        return await run_majority_voting(task, candidates, k, args)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================
# TrustRoute 及其消融版本
# ============================================

async def run_trustroute(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """完整的 TrustRoute"""
    start_time = time.time()
    total_cost = 0.0
    
    # 使用完整的排名机制
    ranked = rank_agents_by_rep_cost(
        candidates,
        w_q=getattr(args, 'w_q', 1.0),
        w_r=getattr(args, 'w_r', 0.3),
        w_c=getattr(args, 'w_c', 0.2),
        min_rep=getattr(args, 'min_rep', 0.0)
    )
    
    if not ranked:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": "No qualified agents"
        }
    
    agent = ranked[0]
    prompt = task.get("prompt", "")
    
    try:
        result: AgentResult = await agent.generate(prompt)
        update_reputation(agent.model, success=True)
        
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time,
            "agent_used": agent.model
        }
    except Exception as e:
        update_reputation(agent.model, success=False)
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


async def run_trustroute_norep(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """TrustRoute 无信誉机制（消融实验）"""
    start_time = time.time()
    
    # 只考虑质量和成本，不考虑信誉
    scored = []
    for agent in candidates:
        cost_norm = agent.cpk / 0.1
        quality = agent.extra.get("quality", 0.7)
        score = 1.0 * quality - 0.2 * cost_norm  # 固定权重
        scored.append((score, agent))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    agent = scored[0][1]
    
    try:
        result: AgentResult = await agent.generate(task.get("prompt", ""))
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


async def run_trustroute_nocost(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """TrustRoute 无成本排序（消融实验）"""
    start_time = time.time()
    
    # 只考虑质量和信誉，忽略成本
    scored = []
    for agent in candidates:
        rep = get_agent_reputation(agent.model)
        quality = agent.extra.get("quality", 0.7)
        score = 1.0 * quality + 0.3 * rep  # 忽略成本项
        scored.append((score, agent))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    agent = scored[0][1]
    
    try:
        result: AgentResult = await agent.generate(task.get("prompt", ""))
        update_reputation(agent.model, success=True)
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time
        }
    except Exception as e:
        update_reputation(agent.model, success=False)
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


# ============================================
# 基础 Baseline 方法
# ============================================

async def run_single_agent(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """Single-Agent baseline（使用第一个/最好的模型）"""
    start_time = time.time()
    
    # 使用候选列表中的第一个模型
    agent = candidates[0]
    
    try:
        result: AgentResult = await agent.generate(task.get("prompt", ""))
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time,
            "agent_used": agent.model
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }


async def run_self_consistency(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    k: int
) -> Dict[str, Any]:
    """Self-Consistency baseline (vote on extracted final answer key, not full text)."""
    import re
    from collections import Counter

    start_time = time.time()
    agent = candidates[0]
    ds = (task.get("dataset") or "").lower()

    # 1) prompt：GSM8K 强制 ####
    prompt = task.get("prompt", "") or task.get("question", "") or ""
    if ds == "gsm8k":
        prompt = prompt.rstrip() + "\n\nPlease solve step-by-step and output the final answer as: #### <number>"

    # 2) vote key：提取最后的“数字/分数”，用于投票
    numpat = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?")
    def vote_key(text: str) -> str:
        if not text:
            return "NO_ANSWER"
        tail = text.split("####")[-1] if "####" in text else text[-250:]
        nums = numpat.findall(tail)
        if not nums:
            return "NO_ANSWER"
        key = nums[-1].replace(",", "")
        key = re.sub(r"\s+", "", key)  # 统一 1 / 2 -> 1/2
        return key.strip()

    answers = []
    keys = []
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # 3) 逐次采样（保持你现在的 latency 口径：串行求和）
    for _ in range(k):
        try:
            result: AgentResult = await agent.generate(prompt, temperature=0.7)
            txt = (result.text or "").strip()
            if not txt:
                continue
            answers.append(txt)
            keys.append(vote_key(txt))

            total_cost += float(getattr(result, "cost_usd", 0.0) or 0.0)
            total_prompt_tokens += int(getattr(result, "prompt_tokens", 0) or 0)
            total_completion_tokens += int(getattr(result, "completion_tokens", 0) or 0)
        except Exception:
            continue

    if not answers:
        return {
            "candidate": "",
            "cost_usd": total_cost,
            "latency_s": time.time() - start_time,
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "agent_used": agent.model,
            "note": f"SC-{k}-all_failed"
        }

    # 4) 多数投票（按 key）
    cnt = Counter(keys)
    best_key, best_votes = cnt.most_common(1)[0]
    # 多个同票，取最后一个（通常更像“最终答案”）
    idxs = [i for i, kk in enumerate(keys) if kk == best_key]
    chosen_idx = idxs[-1]
    final_answer = answers[chosen_idx]

    return {
        "candidate": final_answer,
        "cost_usd": total_cost,
        "latency_s": time.time() - start_time,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "agent_used": agent.model,
        "note": f"SC-{k} vote={best_votes}/{len(keys)} key={best_key}"
    }



# ============================================
# 高级 Baseline 方法（可选实现）
# ============================================

async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    import time
    start = time.time()
    
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    total_cost = 0
    best_result = None
    best_score = -1
    
    # ✅ 添加数据集特定的 prompt 处理
    base_prompt = task.get("prompt", "")
    dataset = task.get("dataset", "")
    
    for agent in sorted_agents:
        # 根据数据集调整 prompt
        prompt = base_prompt
        if dataset == "gsm8k":
            prompt += "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
        
        result = await agent.generate(prompt)  # ✅ 使用增强的 prompt


def compute_quality_score(text: str, dataset: str) -> float:
    """
    启发式质量评分（0-1之间）
    基于多个指标的加权组合
    """
    score = 0.0
    
    if dataset in ["mbpp", "humaneval"]:
        # 代码任务的质量指标
        has_def = "def " in text
        has_return = "return" in text or "yield" in text
        has_docstring = '"""' in text or "'''" in text
        reasonable_length = 50 < len(text) < 2000
        no_syntax_error = check_basic_syntax(text)
        
        score = (
            0.3 * has_def +
            0.3 * has_return +
            0.1 * has_docstring +
            0.2 * reasonable_length +
            0.1 * no_syntax_error
        )
    
    elif dataset == "gsm8k":
        # 数学任务的质量指标（修复：降低对格式的依赖）
        import re
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        has_final_number = bool(numbers)  # 只要有数字就行
        has_calculation = any(op in text for op in ['+', '-', '*', '/', '=', 'x'])
        has_steps = text.count('\n') >= 2
        reasonable_length = 50 < len(text) < 1000
        
        # ✅ 新评分标准（降低对 #### 的依赖）
        score = (
            0.3 * has_final_number +    # 30%: 有最终数字
            0.3 * has_calculation +      # 30%: 有计算过程
            0.2 * has_steps +            # 20%: 有步骤
            0.2 * reasonable_length      # 20%: 长度合理
        )
    
    return min(score, 1.0)


def check_basic_syntax(code: str) -> bool:
    """检查代码是否有明显语法错误"""
    try:
        import ast
        ast.parse(code)
        return True
    except:
        return False


async def run_routellm(
    task: Dict[str, Any],
    candidates: List[BaseAgent],
    args: Any
) -> Dict[str, Any]:
    """RouteLLM baseline（简化实现）"""
    # TODO: 如果要实现完整的RouteLLM，需要训练胜率预测器
    # 这里提供简化版本：基于查询长度的启发式路由
    start_time = time.time()
    
    prompt = task.get("prompt", "")
    prompt_length = len(prompt.split())
    
    # 简单规则：短查询用便宜模型，长查询用贵模型
    if prompt_length < 50:
        agent = min(candidates, key=lambda a: a.cpk)
    else:
        agent = max(candidates, key=lambda a: a.extra.get("quality", 0.5))
    
    try:
        result: AgentResult = await agent.generate(prompt)
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start_time,
            "agent_used": agent.model
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start_time,
            "error": str(e)
        }
async def run_oracle(task: dict, candidates: list, args) -> dict:
    """
    Oracle: 总是选择最强的模型（理论上界）
    
    假设：成本最高的模型 = 能力最强的模型
    """
    import time
    start = time.time()
    
    # ✅ 基于成本选择最强模型
    def get_model_cost(agent):
        """计算模型的平均成本"""
        # 尝试从不同的属性获取成本信息
        if hasattr(agent, 'cpk') and agent.cpk > 0:
            return agent.cpk
        
        # 尝试从 pricing 元数据
        if hasattr(agent, 'extra') and isinstance(agent.extra, dict):
            pricing = agent.extra.get('pricing', {})
            if pricing:
                cost_in = pricing.get('input', 0)
                cost_out = pricing.get('output', 0)
                return (cost_in + cost_out) / 2
        
        # 尝试直接获取价格属性
        cost_in = getattr(agent, 'prompt_price_per_1k', 0)
        cost_out = getattr(agent, 'completion_price_per_1k', 0)
        
        return (cost_in + cost_out) / 2
    
    # 选择成本最高的模型
    best_agent = max(candidates, key=get_model_cost)
    
    # 打印调试信息
    model_costs = [(getattr(a, 'model', str(a)), get_model_cost(a)) for a in candidates]
    print(f"[Oracle] Model costs: {model_costs}")
    print(f"[Oracle] Selected: {getattr(best_agent, 'model', str(best_agent))} "
          f"(cost: ${get_model_cost(best_agent):.6f}/1k)")
    
    # 只调用这一个模型
    result = await best_agent.generate(task.get("prompt", ""))
    
    return {
        "candidate": result.text,
        "cost_usd": result.cost_usd,
        "latency_s": time.time() - start,
        "agent_used": getattr(best_agent, 'model', str(best_agent)),
        "note": "Oracle"
    }


async def run_random_routing(task: dict, candidates: list, args) -> dict:
    """随机选择一个模型（理论下界）"""
    import random, time
    start = time.time()
    
    agent = random.choice(candidates)
    
    # ✅ 添加异常处理
    try:
        result = await agent.generate(task.get("prompt", ""))
        return {
            "candidate": result.text,
            "cost_usd": result.cost_usd,
            "latency_s": time.time() - start,
            "agent_used": agent.model,
            "note": "Random"
        }
    except Exception as e:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": str(e)
        }


async def run_cascade(task: dict, candidates: list, args) -> dict:
    """
    级联路由：从便宜到贵依次尝试，直到答案看起来valid
    """
    import time
    start = time.time()
    
    # 按成本排序
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    total_cost = 0
    last_result = None
    
    for agent in sorted_agents:
        try:
            result = await agent.generate(task.get("prompt", ""))
            total_cost += result.cost_usd
            last_result = result
            
            # ✅ 简单的valid检查
            if is_valid_answer(result.text, task.get("dataset")):
                return {
                    "candidate": result.text,
                    "cost_usd": total_cost,
                    "latency_s": time.time() - start,
                    "agent_used": agent.model,
                    "note": f"Cascade-Success-at-{agent.model}"
                }
        except Exception as e:
            continue
    
    # ✅ 如果都不valid或都失败，返回最后一个
    if last_result:
        return {
            "candidate": last_result.text,
            "cost_usd": total_cost,
            "latency_s": time.time() - start,
            "agent_used": sorted_agents[-1].model if sorted_agents else "unknown",
            "note": "Cascade-Fallback"
        }
    else:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": "All agents failed"
        }


async def run_majority_voting(task: dict, candidates: list, k: int, args) -> dict:
    """
    多数投票：并行调用k个模型，选择最高票的答案
    """
    import time, random
    from collections import Counter
    start = time.time()
    
    # 随机选择k个不同的模型
    selected = random.sample(candidates, min(k, len(candidates)))
    
    # ✅ 修复：添加异常处理
    results = await asyncio.gather(*[
        agent.generate(task.get("prompt", ""))
        for agent in selected
    ], return_exceptions=True)
    
    # ✅ 过滤异常结果
    valid_results = [r for r in results if not isinstance(r, Exception)]
    
    if not valid_results:
        return {
            "candidate": "",
            "cost_usd": 0,
            "latency_s": time.time() - start,
            "error": "All agents failed"
        }
    
    # 提取答案并投票
    answers = [extract_final_answer(r.text, task.get("dataset")) for r in valid_results]
    answer_counts = Counter(answers)
    most_common_answer, count = answer_counts.most_common(1)[0]
    
    # 找到第一个给出这个答案的结果
    chosen_result = None
    for r, ans in zip(valid_results, answers):
        if ans == most_common_answer:
            chosen_result = r
            break
    
    # ✅ 防御性检查
    if chosen_result is None:
        chosen_result = valid_results[0]
    
    total_cost = sum(r.cost_usd for r in valid_results)
    
    return {
        "candidate": chosen_result.text,
        "cost_usd": total_cost,
        "latency_s": time.time() - start,
        "agent_used": f"MajorityVoting-k{k}",
        "note": f"Votes:{count}/{len(valid_results)}"
    }


# 辅助函数
def is_valid_answer(text: str, dataset: str) -> bool:
    """检查答案是否看起来valid（不需要ground truth）"""
    if dataset in ["mbpp", "humaneval"]:
        # 代码任务：必须包含函数定义
        return "def " in text and ("return" in text or "yield" in text)
    elif dataset == "gsm8k":
        # 数学任务：必须包含最终答案标记
        return "####" in text or any(char.isdigit() for char in text[-50:])
    return len(text.strip()) > 20


def extract_final_answer(text: str, dataset: str) -> str:
    """提取最终答案用于投票"""
    if dataset == "gsm8k":
        if "####" in text:
            return text.split("####")[-1].strip()
    elif dataset in ["mbpp", "humaneval"]:
        # 提取函数签名作为答案标识
        import re
        match = re.search(r'def\s+(\w+)\s*\(', text)
        if match:
            return match.group(1)
    return text[:100]  # 默认取前100字符


def load_ground_truth(task: dict) -> str:
    """加载ground truth（仅Oracle使用）"""
    dataset = task.get("dataset")
    if dataset == "gsm8k":
        # GSM8K的答案在原始数据中
        return task.get("answer", "")
    elif dataset in ["mbpp", "humaneval"]:
        # 代码任务的测试用例
        return task.get("test", "")
    return ""


def check_correctness(answer: str, ground_truth: str, dataset: str) -> bool:
    """检查答案是否正确（仅Oracle使用）"""
    if dataset == "gsm8k":
        # 提取数字答案
        import re
        pred_nums = re.findall(r'-?\d+\.?\d*', answer.split("####")[-1] if "####" in answer else answer)
        gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth)
        if pred_nums and gt_nums:
            return abs(float(pred_nums[-1]) - float(gt_nums[-1])) < 0.01
    elif dataset in ["mbpp", "humaneval"]:
        # 代码任务需要运行测试（这里简化为包含关键代码结构）
        return "def " in answer and "return" in answer
    return False

# ================= REPAIRED FUNCTIONS INJECTED BY FIX SCRIPT =================

async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    """
    FrugalGPT (Robust Version): 
    按成本排序，依次尝试。如果模型失败或超时，立即切换下一个。
    """
    import time
    import asyncio
    start = time.time()
    
    # 按成本排序 (便宜 -> 贵)
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    # 数据集特定的 Prompt 后缀
    base_prompt = task.get("prompt", "")
    dataset = task.get("dataset", "")
    suffix = ""
    if dataset == "gsm8k":
        suffix = "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
    
    full_prompt = base_prompt + suffix
    
    last_error = None
    
    for agent in sorted_agents:
        try:
            # 尝试生成
            # print(f"  [FrugalGPT] Trying {agent.model}...")
            result = await agent.generate(full_prompt)
            
            # 检查结果是否为空
            if not result.text or not result.text.strip():
                raise ValueError("Empty response from agent")
                
            # 简单验证 (Cascade 逻辑的一部分，FrugalGPT 也可以用)
            # 如果看起来是合法的（比如有代码或数字），就直接接受
            # 这里为了省钱，只要不报错且不为空，我们就接受 (或者你可以加 is_valid_answer 判断)
            
            if is_valid_answer(result.text, dataset):
                return {
                    "candidate": result.text,
                    "cost_usd": result.cost_usd,
                    "latency_s": time.time() - start,
                    "agent_used": agent.model,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "note": "Frugal-Success"
                }
            else:
                # 结果格式不对，视为失败，尝试下一个昂贵模型
                # print(f"  [FrugalGPT] {agent.model} output invalid, escalating...")
                continue
                
        except Exception as e:
            last_error = e
            # print(f"  [FrugalGPT] {agent.model} failed: {e}, trying next...")
            continue
            
    # 如果所有模型都失败了，返回最后一个异常
    return {
        "candidate": "",
        "cost_usd": 0,
        "latency_s": time.time() - start,
        "error": f"All agents failed. Last error: {str(last_error)}"
    }


async def run_cascade(task: dict, candidates: list, args) -> dict:
    """
    Cascade (Robust Version):
    依次尝试，直到 is_valid_answer 为真。遇到超时直接跳过。
    """
    import time
    start = time.time()
    
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    total_cost = 0
    last_result = None
    last_error = None
    
    dataset = task.get("dataset", "")
    prompt = task.get("prompt", "")
    
    for agent in sorted_agents:
        try:
            result = await agent.generate(prompt)
            total_cost += result.cost_usd
            
            # 检查是否为空
            if not result.text or not result.text.strip():
                continue
                
            last_result = result
            
            # 有效性检查
            if is_valid_answer(result.text, dataset):
                return {
                    "candidate": result.text,
                    "cost_usd": total_cost,
                    "latency_s": time.time() - start,
                    "agent_used": agent.model,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "note": f"Cascade-Success-at-{agent.model}"
                }
        except Exception as e:
            last_error = e
            continue
    
    # 如果都失败了，但在过程中有产生过（无效）结果，返回最后一个
    if last_result:
        return {
            "candidate": last_result.text,
            "cost_usd": total_cost,
            "latency_s": time.time() - start,
            "agent_used": sorted_agents[-1].model if sorted_agents else "unknown",
            "prompt_tokens": last_result.prompt_tokens,
            "completion_tokens": last_result.completion_tokens,
            "note": "Cascade-Fallback-Invalid"
        }
    else:
        return {
            "candidate": "",
            "cost_usd": total_cost,
            "latency_s": time.time() - start,
            "error": f"All agents failed. Last error: {str(last_error)}"
        }




# [PATCHED BY AUTO-SCRIPT]
async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    """
    FrugalGPT (Robust Fixed): 
    按成本排序。如果模型超时/空结果，立即尝试下一个，绝不崩溃。
    """
    import time
    import asyncio
    start = time.time()
    
    # 1. 按成本排序
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    # 2. 准备 Prompt (GSM8K特化)
    base_prompt = task.get("prompt", "")
    dataset = task.get("dataset", "")
    suffix = ""
    if dataset == "gsm8k":
        suffix = "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
    full_prompt = base_prompt + suffix
    
    last_error = None
    
    for agent in sorted_agents:
        try:
            # print(f"  [Frugal] Trying {agent.model}...")
            # 3. 调用模型
            result = await agent.generate(full_prompt)
            
            # 4. 检查是否为空 (关键修复点!)
            if not result.text or not result.text.strip():
                # print(f"  [Frugal] {agent.model} returned empty, skipping...")
                continue
            
            # 5. 只要有内容，就视为成功 (交给后续评价指标去判断对错)
            return {
                "candidate": result.text,
                "cost_usd": result.cost_usd,
                "latency_s": time.time() - start,
                "agent_used": agent.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "note": "Frugal-Success"
            }

        except Exception as e:
            last_error = e
            # print(f"  [Frugal] {agent.model} error: {e}, skipping...")
            continue
            
    # 6. 全军覆没兜底
    return {
        "candidate": "",
        "cost_usd": 0,
        "latency_s": time.time() - start,
        "error": f"All agents failed. Last error: {str(last_error)}"
    }




# [INJECTED ROBUST FIX]
async def run_frugal_gpt(task: dict, candidates: list, args) -> dict:
    """FrugalGPT (Robust): Auto-retry on failure, never crash."""
    import time, asyncio
    start = time.time()
    
    # 按成本排序
    sorted_agents = sorted(candidates, key=lambda a: a.cpk)
    
    # GSM8K Prompt patch
    suffix = ""
    if task.get("dataset") == "gsm8k":
        suffix = "\n\nPlease solve step-by-step and output the final answer as: #### <number>"
    full_prompt = (task.get("prompt", "") or "") + suffix
    
    last_error = None
    
    for agent in sorted_agents:
        try:
            # print(f"Trying {agent.model}...")
            result = await agent.generate(full_prompt)
            
            # 关键：检查是否为空
            if not result.text or not result.text.strip():
                continue # 空结果，跳过，找下一个模型
            
            return {
                "candidate": result.text,
                "cost_usd": result.cost_usd,
                "latency_s": time.time() - start,
                "agent_used": agent.model,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "note": "Frugal-Success"
            }
        except Exception as e:
            last_error = e
            # 出错不崩溃，继续循环找下一个模型
            continue
            
    # 全失败时的兜底
    return {
        "candidate": "",
        "cost_usd": 0,
        "latency_s": time.time() - start,
        "error": f"All agents failed. Last: {str(last_error)}"
    }

