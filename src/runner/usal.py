"""
USAL: Unsupervised Self-Adaptive Learning for LLM Routing
完全Train-Free的信誉更新机制
"""
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Tuple


class USALReputationUpdater:
    """
    USAL的三组件信誉更新器
    """
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        # 记录每个模型的历史表现（用于一致性检查）
        self.history = defaultdict(lambda: deque(maxlen=window_size))
    
    def update(self, model_id: str, task: dict, answer: str, 
               other_answers: Dict[str, str]) -> float:
        """
        综合三个组件计算信誉增量
        
        Args:
            model_id: 模型标识
            task: 任务信息
            answer: 该模型的答案
            other_answers: 其他模型的答案 {model_id: answer}
        
        Returns:
            信誉增量 (-1.0 到 +1.0)
        """
        # 组件1: Proxy Task Validation (权重0.3)
        proxy_score = self._proxy_task_score(task, answer)
        
        # 组件2: Adversarial Cross-Validation (权重0.4)
        adversarial_score = self._adversarial_score(
            task, answer, other_answers
        )
        
        # 组件3: Confidence-Based Online Learning (权重0.3)
        confidence_score = self._confidence_score(
            model_id, task, answer
        )
        
        # 加权组合
        total_score = (
            0.3 * proxy_score +
            0.4 * adversarial_score +
            0.3 * confidence_score
        )
        
        # 记录历史
        self.history[model_id].append({
            "task_type": task.get("dataset"),
            "proxy": proxy_score,
            "adversarial": adversarial_score,
            "confidence": confidence_score,
            "total": total_score
        })
        
        # 归一化到[-0.5, 0.5]
        return total_score - 0.5
    
    def _proxy_task_score(self, task: dict, answer: str) -> float:
        """
        组件1: Proxy Task Validation
        生成可验证的代理任务并测试
        """
        dataset = task.get("dataset")
        score = 0.0
        tests = 0
        
        if dataset in ["mbpp", "humaneval"]:
            # 代码任务的代理测试
            tests = 5
            
            # Test 1: 是否包含函数定义
            score += 1.0 if "def " in answer else 0.0
            
            # Test 2: 是否包含return语句
            score += 1.0 if ("return" in answer or "yield" in answer) else 0.0
            
            # Test 3: 语法检查
            try:
                import ast
                ast.parse(answer)
                score += 1.0
            except:
                pass
            
            # Test 4: 是否处理边界情况（通过注释或条件判断推断）
            score += 0.5 if "if" in answer else 0.0
            score += 0.5 if ("None" in answer or "[]" in answer) else 0.0
            
            # Test 5: 代码复杂度合理性
            lines = answer.count('\n')
            score += 1.0 if 5 <= lines <= 50 else 0.0
        
        elif dataset == "gsm8k":
            # 数学任务的代理测试
            tests = 4
            
            # Test 1: 是否包含最终答案标记
            score += 1.0 if "####" in answer else 0.0
            
            # Test 2: 是否包含数字
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            score += 1.0 if numbers else 0.0
            
            # Test 3: 是否有步骤（多行）
            score += 1.0 if answer.count('\n') >= 2 else 0.0
            
            # Test 4: 数字合理性检查（不能太大或太小）
            if numbers:
                try:
                    final_num = float(numbers[-1])
                    score += 1.0 if 0.01 <= abs(final_num) <= 1e6 else 0.0
                except:
                    pass
        
        return score / tests if tests > 0 else 0.5
    
    def _adversarial_score(self, task: dict, answer: str, 
                          other_answers: Dict[str, str]) -> float:
        """
        组件2: Adversarial Cross-Validation
        模拟其他模型对当前答案的挑战
        """
        if not other_answers:
            return 0.5  # 没有其他模型时返回中性分数
        
        # 一致性得分：与其他模型的答案相似度
        consistency_score = 0.0
        
        # 提取当前答案的关键特征
        my_features = self._extract_features(answer, task.get("dataset"))
        
        agree_count = 0
        for other_answer in other_answers.values():
            other_features = self._extract_features(
                other_answer, task.get("dataset")
            )
            
            # 特征匹配度
            similarity = self._feature_similarity(my_features, other_features)
            if similarity > 0.6:  # 阈值
                agree_count += 1
        
        # 多数同意 → 高分；少数孤立 → 低分
        consensus = agree_count / len(other_answers)
        
        # 但也要奖励独特但高质量的答案（避免简单从众）
        quality_bonus = self._answer_quality(answer, task.get("dataset"))
        
        # 综合评分
        return 0.7 * consensus + 0.3 * quality_bonus
    
    def _confidence_score(self, model_id: str, task: dict, 
                         answer: str) -> float:
        """
        组件3: Confidence-Based Online Learning
        基于历史表现和当前输出的置信度
        """
        # 1. 当前答案的置信度
        current_confidence = self._answer_quality(answer, task.get("dataset"))
        
        # 2. 与历史表现的一致性
        history_consistency = self._check_history_consistency(
            model_id, task.get("dataset"), current_confidence
        )
        
        # 综合得分
        return 0.6 * current_confidence + 0.4 * history_consistency
    
    def _extract_features(self, answer: str, dataset: str) -> dict:
        """提取答案特征用于比较"""
        features = {}
        
        if dataset in ["mbpp", "humaneval"]:
            features["has_def"] = "def " in answer
            features["has_return"] = "return" in answer
            features["num_lines"] = answer.count('\n')
            features["length"] = len(answer)
            
            # 提取函数签名
            import re
            match = re.search(r'def\s+(\w+)\s*\((.*?)\)', answer)
            if match:
                features["func_name"] = match.group(1)
                features["params"] = match.group(2)
        
        elif dataset == "gsm8k":
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            features["has_final_answer"] = "####" in answer
            features["num_steps"] = answer.count('\n')
            features["final_number"] = numbers[-1] if numbers else None
        
        return features
    
    def _feature_similarity(self, f1: dict, f2: dict) -> float:
        """计算特征相似度"""
        if not f1 or not f2:
            return 0.0
        
        common_keys = set(f1.keys()) & set(f2.keys())
        if not common_keys:
            return 0.0
        
        matches = sum(1 for k in common_keys if f1[k] == f2[k])
        return matches / len(common_keys)
    
    def _answer_quality(self, answer: str, dataset: str) -> float:
        """评估答案的内在质量（不依赖其他模型）"""
        # 复用之前FrugalGPT的quality_score函数
        from .executor import compute_quality_score
        return compute_quality_score(answer, dataset)
    
    def _check_history_consistency(self, model_id: str, 
                                   task_type: str, 
                                   current_score: float) -> float:
        """检查与历史表现的一致性"""
        if model_id not in self.history or len(self.history[model_id]) < 3:
            return 0.5  # 历史不足时返回中性值
        
        # 计算同类任务的历史平均分
        similar_tasks = [
            h for h in self.history[model_id]
            if h["task_type"] == task_type
        ]
        
        if not similar_tasks:
            return 0.5
        
        historical_avg = np.mean([h["total"] for h in similar_tasks])
        
        # 当前得分与历史平均的接近程度
        diff = abs(current_score - historical_avg)
        
        # 差异越小，一致性越高
        consistency = max(0, 1.0 - diff)
        
        return consistency


# 全局单例
_usal_updater = USALReputationUpdater()


def update_reputation_usal(model_id: str, task: dict, answer: str,
                           other_answers: Dict[str, str]) -> float:
    """
    对外接口：使用USAL更新信誉
    
    Returns:
        信誉增量
    """
    return _usal_updater.update(model_id, task, answer, other_answers)