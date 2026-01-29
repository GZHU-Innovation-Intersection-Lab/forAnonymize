import re
import time
import asyncio
import ast
from typing import Dict, List, Any, Callable, Tuple, Optional
from collections import Counter
from .lite_utils import (
    load_rep_state,
    save_rep_state,
    get_rep,
    set_rep,
    rank_agents_by_rep_cost
)


MCQ_DATASETS = {"hellaswag", "arc_challenge", "winogrande", "mmlu"}

FIX_VERSION = "2025-12-16-CONFIDENCE_HOTFIX"
print(f"[ours_lite_v2] LOADED FIX_VERSION={FIX_VERSION}")

def extract_and_repair_code(text: str) -> str:
    """
    从 LLM 输出中提取代码，并进行基本修复
    """
    # 1. 尝试提取 markdown 代码块
    code_block_pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        # 取最长的代码块
        code = max(matches, key=len)
    else:
        # 没有代码块，尝试提取 def 开头的部分
        lines = text.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def '):
                in_function = True
            
            if in_function:
                code_lines.append(line)
                
                # 检测函数结束（下一个非缩进行）
                if line and not line[0].isspace() and code_lines and line != code_lines[0]:
                    break
        
        code = '\n'.join(code_lines) if code_lines else text
    
    # 2. 基本清理
    code = code.strip()
    
    # 3. 移除常见的无关文本
    if not code.startswith('def '):
        # 查找第一个 def
        match = re.search(r'(def\s+\w+.*)', code, re.DOTALL)
        if match:
            code = match.group(1)
    
    # 4. 尝试语法修复
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        code = _attempt_syntax_repair(code)
    
    return code


def _attempt_syntax_repair(code: str) -> str:
    """
    尝试修复常见的语法错误
    """
    # 1. 修复未闭合的括号
    open_parens = code.count('(')
    close_parens = code.count(')')
    if open_parens > close_parens:
        code += ')' * (open_parens - close_parens)
    
    # 2. 修复未闭合的引号
    single_quotes = code.count("'")
    if single_quotes % 2 == 1:
        code += "'"
    
    double_quotes = code.count('"')
    if double_quotes % 2 == 1:
        code += '"'
    
    # 3. 确保至少有 return 语句
    if 'def ' in code and 'return' not in code.lower():
        lines = code.split('\n')
        for i in range(len(lines)-1, -1, -1):
            if lines[i].strip() and lines[i][0].isspace():
                indent = len(lines[i]) - len(lines[i].lstrip())
                lines.insert(i+1, ' ' * indent + 'return None')
                break
        code = '\n'.join(lines)
    
    return code


# ============================================
# 轻量级测试
# ============================================

def run_light_tests(code: str, task: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Light tests:
    - MBPP: run first few assert lines (from test_list if available)
    - HumanEval: exec the whole test string, then call check(candidate)
    - GSM8K: only do format check (has number), but DO NOT treat it as tests_ok
    """
    dataset = task.get("dataset", "").lower()

    # 1) syntax
    try:
        ast.parse(code)
    except SyntaxError:
        return False, "syntax_error"

    # 2) exec solution
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f"exec_error:{type(e).__name__}"

    # 3) code tasks
    if dataset in ["mbpp", "humaneval"]:
        # pick entry function
        entry = task.get("entry_point") or ""
        func_name = None
        if entry and entry in namespace and callable(namespace[entry]):
            func_name = entry
        else:
            funcs = [name for name, obj in namespace.items()
                     if callable(obj) and not name.startswith("_")]
            if not funcs:
                return False, "no_function"
            func_name = funcs[0]

        namespace["candidate"] = namespace[func_name]

        # MBPP-style: list of assert lines
        test_list = task.get("test_list", None)
        if isinstance(test_list, list) and test_list:
            tests = [t for t in test_list if isinstance(t, str) and t.strip()]
            for i, test in enumerate(tests[:3]):
                test_code = test.strip().replace("candidate", func_name)
                try:
                    exec(test_code, namespace)
                except AssertionError:
                    return False, f"assert_fail_{i}"
                except Exception as e:
                    return False, f"test_error_{i}:{type(e).__name__}"
            return True, "pass"

        # HumanEval-style: one big test program
        test_prog = task.get("test", "")
        if isinstance(test_prog, str) and test_prog.strip():
            try:
                exec(test_prog, namespace)
                if "check" in namespace and callable(namespace["check"]):
                    namespace["check"](namespace["candidate"])
                return True, "pass"
            except AssertionError:
                return False, "assert_fail"
            except Exception as e:
                return False, f"test_error:{type(e).__name__}"

        return True, "exec_pass"

    # 4) GSM8K: format hint only
    if dataset == "gsm8k":
        has_number = bool(re.search(r"\d", code))
        return (has_number, "format_ok" if has_number else "no_number")

    return True, "exec_pass"



# ============================================
# 主函数 run_ours_lite
# ============================================

async def run_ours_lite(
    task: Dict[str, Any],
    candidate_agents: List[Dict[str, Any]],
    generate_fn: Callable, 
    judge_fn: Optional[Callable] = None,
    budget_usd: float = 0.20,
    tau1: float = 0.95,
    tau2: float = 0.80,
    tau2m: float = 0.20,
    max_k: int = 3,
    max_retries: int = 1,
    cross_tier1: bool = False,
    eta_rep: float = 0.3,
    enable_short_code_prompt: bool = True,
    enable_light_tests: bool = True,
    disable_cost_ranking: bool = False,
    disable_reputation: bool = False,
    disable_diversity: bool = False,
) -> Dict[str, Any]:
    """
    TrustRoute with USAL (Unsupervised Self-Adaptive Learning)
    """
    
    print(f"\n{'='*60}")
    print(f"[TrustRoute-USAL] Starting task: {task.get('task_id', 'unknown')}")
    print(f"{'='*60}")
    
    t0 = time.time()
    dataset = (task.get("dataset") or "").lower()
    
    # ============================================
    # 初始化
    # ============================================
    rep_state = load_rep_state()
    
    # 按信誉和成本排序 agents
    agents = candidate_agents[:]
    if not disable_cost_ranking:
        agents = rank_agents_by_rep_cost(agents, rep_state)
    
    if not agents:
        print("[TrustRoute-USAL] ERROR: No agents available!")
        return _pack_result({}, [], 0, t0, "no_agents")
    
    # 准备 prompt 后缀（用于代码任务）
    prompt_suffix = ""
    if enable_short_code_prompt and dataset in ["mbpp", "humaneval"]:
        prompt_suffix = "\n\nPlease provide a concise solution with minimal comments."
    
    # 用于记录所有生成的答案（USAL 需要）
    all_generations = {}
    log = []
    total_cost = 0.0
    
    # ============================================
    # Stage 1: 信誉最高的单个模型
    # ============================================
    print(f"\n[Stage 1] Using top agent: {agents[0].get('name')}")
    print(f"          Reputation: {get_rep(rep_state, agents[0].get('name')):.3f}")
    
    agent1 = agents[0]
    agent1_name = agent1.get("name")
    
    # 生成答案
    res1 = await generate_fn(agent1, task, prompt_suffix, temperature=0.2)
    total_cost += res1.get("cost_usd", 0)
    
    answer1 = res1.get("text", "")
    all_generations[agent1_name] = answer1
    
    # 提取代码（如果是代码任务）
    code1 = extract_and_repair_code(answer1) if dataset in ["mbpp", "humaneval"] else answer1
    
    # 轻量级测试（可选）
    tests_ok = False
    tests_msg = "no_tests"
    if enable_light_tests:
        ok, msg = run_light_tests(code1, task)
        tests_msg = msg
        if dataset in ["mbpp", "humaneval"]:
            tests_ok = (ok and msg == "pass")
        else:
            tests_ok = False  # ✅ gsm8k 永远不要把 pot_valid 当 tests_ok
        print(f"          Test result: {msg}")

    # =========================================================================
    # ✅ FIX APPLIED HERE: Compute stage-1 confidence BEFORE accessing it
    # =========================================================================
    confidence1 = _compute_confidence(code1, dataset, tests_ok)
    # =========================================================================
    
    # 记录候选答案
    cand1 = {
        "agent": agent1_name,
        "text": answer1,
        "code": code1,
        "tests_ok": tests_ok,
        "confidence": confidence1,
        "prompt_tokens": res1.get("prompt_tokens", 0),
        "completion_tokens": res1.get("completion_tokens", 0),
        "cost_usd": res1.get("cost_usd", 0)
    }
    log.append(cand1)
    
    # 动态调整 tau1 阈值
    top_rep = get_rep(rep_state, agent1_name)
    tau1_adjusted = tau1 * (1.0 - 0.1 * min(top_rep / 100.0, 1.0))
    
    # 如果置信度够高，直接返回（早停）
    if confidence1 >= tau1_adjusted:
        print(f"[Stage 1] ✅ High confidence ({confidence1:.3f} >= {tau1_adjusted:.3f}), early stop!")
        
        if not disable_reputation:
            _update_reputation_usal_single(
                rep_state, agent1_name, task, answer1, 
                tests_ok, eta_rep
            )
            save_rep_state(rep_state)
        
        return _pack_result(cand1, log, total_cost, t0, "stage1_early_stop")
    
    print(f"[Stage 1] ⚠️  Low confidence ({confidence1:.3f} < {tau1_adjusted:.3f}), need more agents")
    
    # ============================================
    # Stage 2: 多模型验证和投票
    # ============================================
    if max_k < 2:
        print("[Stage 2] Skipped (max_k < 2)")
        return _pack_result(cand1, log, total_cost, t0, "stage1_only")
    
    print(f"\n[Stage 2] Calling {max_k-1} additional agents...")
    
    additional_agents = _select_additional_agents(
        agents, agent1_name, max_k - 1, 
        disable_diversity, rep_state
    )
    
    additional_tasks = [
        generate_fn(agent, task, prompt_suffix, temperature=0.7)
        for agent in additional_agents
    ]
    additional_results = await asyncio.gather(*additional_tasks, return_exceptions=True)
    
    for agent, res in zip(additional_agents, additional_results):
        if isinstance(res, Exception):
            print(f"          [ERROR] {agent.get('name')}: {res}")
            continue
        
        agent_name = agent.get("name")
        answer = res.get("text", "")
        cost = res.get("cost_usd", 0)
        total_cost += cost
        
        all_generations[agent_name] = answer
        
        code = extract_and_repair_code(answer) if dataset in ["mbpp", "humaneval"] else answer
        
        tests_ok_i = False
        if enable_light_tests:
            ok, msg = run_light_tests(code, task)
            if dataset in ["mbpp", "humaneval"]:
                tests_ok_i = (ok and msg == "pass")
            else:
                tests_ok_i = False

        
        confidence_i = _compute_confidence(code, dataset, tests_ok_i)
        
        print(f"          {agent_name}: confidence={confidence_i:.3f}, tests_ok={tests_ok_i}")
        
        cand_i = {
            "agent": agent_name,
            "text": answer,
            "code": code,
            "tests_ok": tests_ok_i,
            "confidence": confidence_i,
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
            "cost_usd": cost
        }
        log.append(cand_i)
    
    # ============================================
    # 使用 USAL 更新所有模型的信誉
    # ============================================
    if not disable_reputation:
        print(f"\n[USAL] Updating reputation for {len(all_generations)} agents...")
        _update_reputation_usal_batch(
            rep_state, task, all_generations, log, eta_rep
        )
        save_rep_state(rep_state)
    
    # ============================================
    # 选择最终答案
    # ============================================
    print(f"\n[Final Selection] Choosing best answer from {len(log)} candidates...")
    
    passed_candidates = [c for c in log if c.get("tests_ok", False)]
    if passed_candidates:
        best = max(passed_candidates, key=lambda c: c.get("confidence", 0))
        print(f"              ✅ Selected {best['agent']} (tests passed, conf={best['confidence']:.3f})")
        return _pack_result(best, log, total_cost, t0, "stage2_verified")
    
    if dataset in ["mbpp", "humaneval", "gsm8k"] or dataset in MCQ_DATASETS:
        winner, vote_count, total_votes = _majority_vote(log, dataset)
        agreement_rate = vote_count / total_votes if total_votes > 0 else 0
        
        print(f"              🗳️  Voting: {vote_count}/{total_votes} agree ({agreement_rate:.1%})")
        
        if agreement_rate >= tau2:
            print(f"              ✅ Selected by majority vote: {winner['agent']}")
            return _pack_result(winner, log, total_cost, t0, "stage2_vote")
    
    best = max(log, key=lambda c: c.get("confidence", 0))
    print(f"              ⚠️  Fallback to highest confidence: {best['agent']} ({best['confidence']:.3f})")
    
    return _pack_result(best, log, total_cost, t0, "stage2_confidence")


# ============================================
# 辅助函数
# ============================================

def _compute_confidence(answer: str, dataset: str, tests_ok: bool) -> float:
    # 先算 base_score（跟你原来一样）
    if dataset in ["mbpp", "humaneval"]:
        has_def = "def " in answer
        has_return = "return" in answer or "yield" in answer
        reasonable_length = 50 < len(answer) < 2000
        has_docstring = '"""' in answer or "'''" in answer

        syntax_ok = False
        try:
            import ast
            ast.parse(answer)
            syntax_ok = True
        except:
            pass

        base_score = (
            0.3 * has_def +
            0.3 * has_return +
            0.2 * syntax_ok +
            0.1 * reasonable_length +
            0.1 * has_docstring
        )

        # ✅ 轻量测试通过：小幅加分
        if tests_ok:
            base_score = min(1.0, base_score + 0.10)

    elif dataset == "gsm8k":
        has_final_answer = "####" in answer
        has_numbers = any(char.isdigit() for char in answer)
        has_steps = answer.count('\n') >= 2
        reasonable_length = 50 < len(answer) < 1000

        base_score = (
            0.4 * has_final_answer +
            0.3 * has_numbers +
            0.2 * has_steps +
            0.1 * reasonable_length
        )
        # GSM8K 不用 tests_ok 加分（见 4.3）
    else:
        # MCQ: 有明确选项字母则置信度更高
        if dataset in MCQ_DATASETS:
            import re
            m = re.search(r"(?:^|\b)([ABCD])(?:\b|$)", answer.strip().upper())
            base_score = 0.90 if m else 0.35
        else:
            base_score = 0.5 if len(answer.strip()) > 20 else 0.2


    return min(base_score, 1.0)



def _select_additional_agents(
    agents: List[Dict], 
    exclude_name: str, 
    count: int,
    disable_diversity: bool,
    rep_state: Dict
) -> List[Dict]:
    """
    选择额外的agents用于Stage 2
    """
    candidates = [a for a in agents if a.get("name") != exclude_name]
    
    if disable_diversity:
        return candidates[:count]
    
    # Diversity策略：混合选择
    result = []
    
    # 1. 至少选一个高信誉的
    if candidates:
        result.append(candidates[0])
    
    # 2. 选一个不同tier的
    if len(candidates) > 1:
        # 找到tier不同的
        first_tier = candidates[0].get("tier", "unknown")
        for agent in candidates[1:]:
            if agent.get("tier") != first_tier:
                result.append(agent)
                break
        else:
            # 没找到不同tier的，就选下一个
            result.append(candidates[1])
    
    # 3. 填充剩余
    for agent in candidates:
        if agent not in result:
            result.append(agent)
        if len(result) >= count:
            break
    
    return result[:count]


def _majority_vote(
    candidates: List[Dict], 
    dataset: str
) -> Tuple[Dict, int, int]:
    """
    多数投票选择答案
    """
    # 提取每个候选的"关键答案"用于投票
    votes = []
    for cand in candidates:
        key_answer = _extract_key_answer(cand.get("text", ""), dataset)
        votes.append((key_answer, cand))
    
    # 统计投票
    vote_counts = Counter([v[0] for v in votes])
    
    if not vote_counts:
        return candidates[0], 0, len(candidates)
    
    # 找到得票最多的答案
    most_common_answer, count = vote_counts.most_common(1)[0]
    
    # 返回第一个给出该答案的候选
    for key_answer, cand in votes:
        if key_answer == most_common_answer:
            return cand, count, len(votes)
    
    return candidates[0], 0, len(candidates)


def _extract_key_answer(text: str, dataset: str) -> str:
    """提取用于投票的关键答案"""
    if dataset == "gsm8k":
        # 数学：提取最终数字
        if "####" in text:
            final_part = text.split("####")[-1].strip()
        else:
            final_part = text[-100:]
        
        import re
        numbers = re.findall(r'-?\d+\.?\d*', final_part)
        return numbers[-1] if numbers else "NO_ANSWER"
    
    if dataset in ["mbpp", "humaneval"]:
        # 代码：提取函数签名
        import re
        match = re.search(r'def\s+(\w+)\s*\((.*?)\)', text)
        if match:
            return f"{match.group(1)}({match.group(2)})"
        return "NO_FUNCTION"
    elif dataset in MCQ_DATASETS:
        import re
        # Winogrande 只有 A/B；其他一般 A-D
        allow = "AB" if dataset == "winogrande" else "ABCD"
        t = text.strip().upper()
        # 优先抓 "Answer: X"
        m = re.search(r"ANSWER\s*[:：]\s*([%s])" % allow, t)
        if m:
            return m.group(1)
        # 再抓最后出现的选项字母
        m2 = re.findall(r"(?:^|\b)([%s])(?:\b|$)" % allow, t)
        return m2[-1] if m2 else "NO_ANSWER"

    # 默认：取前100字符
    return text[:100].strip()


def _update_reputation_usal_single(
    rep_state: Dict,
    agent_name: str,
    task: Dict,
    answer: str,
    tests_ok: bool,
    eta: float
):
    """
    单模型的简化USAL更新（Stage 1早停时使用）
    """
    # 1. Proxy Task Score (基于测试)
    proxy_score = 0.85 if tests_ok else 0.3
    
    # 2. Confidence Score (基于答案质量)
    dataset = task.get("dataset", "")
    confidence = _compute_confidence(answer, dataset, tests_ok)
    
    # 综合得分
    total_score = 0.5 * proxy_score + 0.5 * confidence
    
    # 转换为信誉增量 [-5, +10]
    rep_delta = (total_score - 0.5) * 20  # 归一化到[-10, +10]，偏向正向
    
    # 更新信誉
    current_rep = get_rep(rep_state, agent_name)
    new_rep = current_rep + eta * rep_delta
    set_rep(rep_state, agent_name, new_rep)
    
    print(f"          [USAL] {agent_name}: {current_rep:.1f} -> {new_rep:.1f} (Δ{eta*rep_delta:+.2f})")


def _update_reputation_usal_batch(
    rep_state: Dict,
    task: Dict,
    all_answers: Dict[str, str],  # {agent_name: answer}
    candidates: List[Dict],       # 包含tests_ok等信息
    eta: float
):
    """
    批量更新所有模型的信誉（Stage 2使用）
    """
    dataset = task.get("dataset", "")
    
    for agent_name, answer in all_answers.items():
        # 获取该模型的详细信息
        cand_info = next((c for c in candidates if c["agent"] == agent_name), None)
        if not cand_info:
            continue
        
        # 计算USAL得分（简化版，不依赖外部USAL模块）
        # 组件1: 测试得分
        test_score = 1.0 if cand_info.get("tests_ok", False) else 0.0
        
        # 组件2: 一致性得分（与其他模型的答案比较）
        agreement_scores = []
        for other_name, other_answer in all_answers.items():
            if other_name == agent_name:
                continue
            # 简单的文本相似度（可以用更复杂的方法）
            similarity = _compute_answer_similarity(answer, other_answer, dataset)
            agreement_scores.append(similarity)
        
        agreement_score = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
        
        # 组件3: 置信度得分
        confidence_score = cand_info.get("confidence", 0.5)
        
        # 综合得分（加权平均）
        total_score = (
            0.4 * test_score +
            0.3 * agreement_score +
            0.3 * confidence_score
        )
        
        # 转换为信誉增量
        rep_delta = (total_score - 0.5) * 20  # 归一化到[-10, +10]
        
        # 更新信誉
        current_rep = get_rep(rep_state, agent_name)
        new_rep = current_rep + eta * rep_delta
        set_rep(rep_state, agent_name, new_rep)
        
        print(f"          [USAL] {agent_name}: {current_rep:.1f} -> {new_rep:.1f} "
              f"(test={test_score:.2f}, agree={agreement_score:.2f}, conf={confidence_score:.2f})")


def _compute_answer_similarity(ans1: str, ans2: str, dataset: str) -> float:
    """
    计算两个答案的相似度 (0-1)
    """
    # 提取关键答案
    key1 = _extract_key_answer(ans1, dataset)
    key2 = _extract_key_answer(ans2, dataset)
    
    # 完全匹配
    if key1 == key2:
        return 1.0
    
    # 部分匹配（仅用于数学题）
    if dataset == "gsm8k" and key1 != "NO_ANSWER" and key2 != "NO_ANSWER":
        try:
            num1 = float(key1)
            num2 = float(key2)
            # 容忍1%的误差
            if abs(num1 - num2) / max(abs(num1), abs(num2), 1e-6) < 0.01:
                return 0.9
        except:
            pass
    
    # 字符串相似度（Jaccard）
    set1 = set(key1.lower().split())
    set2 = set(key2.lower().split())
    
    if not set1 and not set2:
        return 0.5
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def _pack_result(
    candidate: Dict,
    log: List[Dict],
    total_cost: float,
    start_time: float,
    reason: str
) -> Dict[str, Any]:
    """打包最终返回结果"""
    elapsed = time.time() - start_time
    
    result = {
        "candidate": candidate.get("text", ""),
        "agent_used": candidate.get("agent", "unknown"),
        "cost_usd": total_cost,
        "latency_s": elapsed,
        "reason": reason,
        "confidence": candidate.get("confidence", 0.0),
        "tests_ok": candidate.get("tests_ok", False),
        "num_agents_called": len(log),
        "log": log
    }
    
    print(f"\n{'='*60}")
    print(f"[Result] Agent: {result['agent_used']}")
    print(f"        Cost: ${total_cost:.4f}")
    print(f"        Time: {elapsed:.2f}s")
    print(f"        Reason: {reason}")
    print(f"{'='*60}\n")
    
    return result
