"""
SwarmMind 经验回放系统

从历史执行中学习和优化：
- 记录执行经验 (task, plan, result, review)
- 检索相似任务经验
- 分析成功率和失败原因
"""

import os
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from .vector_memory import VectorMemoryStore, MemoryEntry


class Experience(BaseModel):
    """执行经验"""
    id: str = ""
    task: str                                    # 用户任务描述
    plan: Dict[str, Any] = Field(default_factory=dict)  # 执行计划
    result: str = ""                             # 执行结果
    review_passed: bool = False                  # 审查是否通过
    review_feedback: str = ""                    # 审查反馈
    suggestions: List[str] = Field(default_factory=list)  # 改进建议
    success: bool = False                        # 是否成功完成
    created_at: str = ""                         # 创建时间
    duration_seconds: float = 0.0                # 执行耗时


class ExperienceStore:
    """
    经验回放存储

    功能：
    - 记录每次执行的经验
    - 基于向量相似度检索历史经验
    - 分析任务成功率
    """

    def __init__(self, persist_dir: str):
        """
        初始化经验存储

        参数:
        - persist_dir: 持久化目录
        """
        self.persist_dir = persist_dir
        self.experience_file = os.path.join(persist_dir, "experiences.jsonl")

        # 向量存储
        self.vector_store = VectorMemoryStore(
            persist_dir=os.path.join(persist_dir, "vectors"),
            collection_name="experiences"
        )

        # 内存缓存
        self._experiences: Dict[str, Experience] = {}
        self._load_experiences()

    def _load_experiences(self) -> None:
        """从文件加载经验"""
        if not os.path.exists(self.experience_file):
            return

        try:
            with open(self.experience_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        exp = Experience(**data)
                        self._experiences[exp.id] = exp
        except Exception as e:
            print(f"[ExperienceStore] Load error: {e}")

    def _save_experience(self, experience: Experience) -> None:
        """保存经验到文件"""
        try:
            os.makedirs(os.path.dirname(self.experience_file), exist_ok=True)
            with open(self.experience_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(experience.model_dump(), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[ExperienceStore] Save error: {e}")

    async def record(
        self,
        task: str,
        plan: Dict[str, Any],
        result: str,
        review_passed: bool,
        review_feedback: str = "",
        suggestions: List[str] = None,
        duration_seconds: float = 0.0
    ) -> str:
        """
        记录一次执行经验

        参数:
        - task: 用户任务描述
        - plan: 执行计划
        - result: 执行结果
        - review_passed: 审查是否通过
        - review_feedback: 审查反馈
        - suggestions: 改进建议
        - duration_seconds: 执行耗时

        返回:
        - experience_id: 经验 ID
        """
        import uuid

        exp_id = str(uuid.uuid4())
        experience = Experience(
            id=exp_id,
            task=task,
            plan=plan,
            result=result,
            review_passed=review_passed,
            review_feedback=review_feedback,
            suggestions=suggestions or [],
            success=review_passed,
            created_at=datetime.now().isoformat(),
            duration_seconds=duration_seconds
        )

        # 存储到内存
        self._experiences[exp_id] = experience

        # 持久化
        self._save_experience(experience)

        # 向量化存储（用于语义检索）
        await self._store_to_vector(experience)

        return exp_id

    async def _store_to_vector(self, experience: Experience) -> None:
        """存储到向量数据库"""
        if not self.vector_store.is_available:
            return

        # 构建用于检索的文本
        text = f"Task: {experience.task}\n"
        if experience.plan:
            text += f"Plan: {json.dumps(experience.plan.get('analysis', ''), ensure_ascii=False)}\n"
        text += f"Success: {experience.success}"

        metadata = {
            "type": "experience",
            "task": experience.task[:500],
            "success": experience.success,
            "review_passed": experience.review_passed,
            "created_at": experience.created_at
        }

        await self.vector_store.store(text, metadata)

    async def recall(
        self,
        task: str,
        top_k: int = 3,
        success_only: bool = False
    ) -> List[Experience]:
        """
        检索相似任务的经验

        参数:
        - task: 任务描述
        - top_k: 返回数量
        - success_only: 只返回成功的经验

        返回:
        - List[Experience]: 相似经验列表
        """
        # 先尝试向量检索
        if self.vector_store.is_available:
            entries = await self.vector_store.search(
                query=task,
                top_k=top_k * 2,  # 多取一些，后续过滤
                where_filter={"type": "experience"}
            )

            results = []
            seen_ids = set()

            for entry in entries:
                # 使用 metadata 中的 task 进行精确匹配
                entry_task = entry.metadata.get("task", "") if entry.metadata else ""

                # 从内存中查找完整经验
                for exp_id, exp in self._experiences.items():
                    if exp_id in seen_ids:
                        continue

                    # 精确匹配：task 完全相等或 entry_task 包含在 exp.task 中
                    if exp.task == entry_task or (entry_task and entry_task in exp.task):
                        if not success_only or exp.success:
                            results.append(exp)
                            seen_ids.add(exp_id)
                            break

                if len(results) >= top_k:
                    break

            return results

        # 回退到关键词匹配
        return self._keyword_search(task, top_k, success_only)

    def _keyword_search(
        self,
        task: str,
        top_k: int,
        success_only: bool
    ) -> List[Experience]:
        """关键词搜索"""
        task_lower = task.lower()
        results = []

        for exp in self._experiences.values():
            if success_only and not exp.success:
                continue

            # 简单的关键词匹配
            if any(word in exp.task.lower() for word in task_lower.split()):
                results.append(exp)

        # 按时间倒序排列
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results[:top_k]

    def get_by_id(self, exp_id: str) -> Optional[Experience]:
        """根据 ID 获取经验"""
        return self._experiences.get(exp_id)

    async def get_success_rate(self, task_pattern: str = "") -> float:
        """
        获取任务成功率

        参数:
        - task_pattern: 任务模式匹配（空字符串表示所有任务）

        返回:
        - 成功率 (0.0 - 1.0)
        """
        if not self._experiences:
            return 0.0

        filtered = list(self._experiences.values())

        if task_pattern:
            filtered = [
                exp for exp in filtered
                if task_pattern.lower() in exp.task.lower()
            ]

        if not filtered:
            return 0.0

        success_count = sum(1 for exp in filtered if exp.success)
        return success_count / len(filtered)

    def get_statistics(self) -> Dict[str, Any]:
        """获取经验统计"""
        experiences = list(self._experiences.values())

        if not experiences:
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "success_rate": 0.0,
                "avg_duration": 0.0
            }

        success_count = sum(1 for exp in experiences if exp.success)
        total_duration = sum(exp.duration_seconds for exp in experiences)

        return {
            "total": len(experiences),
            "success": success_count,
            "failed": len(experiences) - success_count,
            "success_rate": success_count / len(experiences),
            "avg_duration": total_duration / len(experiences) if experiences else 0.0
        }

    def get_recent_experiences(self, limit: int = 10) -> List[Experience]:
        """获取最近的经验"""
        experiences = list(self._experiences.values())
        experiences.sort(key=lambda x: x.created_at, reverse=True)
        return experiences[:limit]

    def clear(self) -> int:
        """清空所有经验"""
        count = len(self._experiences)
        self._experiences.clear()

        # 删除文件
        if os.path.exists(self.experience_file):
            try:
                os.remove(self.experience_file)
            except Exception:
                pass

        return count


class ExperienceAnalyzer:
    """经验分析器"""

    def __init__(self, store: ExperienceStore):
        self.store = store

    def analyze_failures(self) -> List[Dict[str, Any]]:
        """分析失败原因"""
        failures = [
            exp for exp in self.store._experiences.values()
            if not exp.success
        ]

        patterns = []

        # 按反馈分组
        feedback_groups: Dict[str, List[Experience]] = {}
        for exp in failures:
            key = exp.review_feedback[:100] if exp.review_feedback else "无反馈"
            if key not in feedback_groups:
                feedback_groups[key] = []
            feedback_groups[key].append(exp)

        for feedback, exps in feedback_groups.items():
            patterns.append({
                "pattern": feedback,
                "count": len(exps),
                "examples": [e.task for e in exps[:3]]
            })

        patterns.sort(key=lambda x: x["count"], reverse=True)
        return patterns

    def get_common_tools(self, success_only: bool = True) -> Dict[str, int]:
        """获取常用工具统计"""
        tool_counts: Dict[str, int] = {}

        for exp in self.store._experiences.values():
            if success_only and not exp.success:
                continue

            if exp.plan and "actions" in exp.plan:
                for action in exp.plan.get("actions", []):
                    tool = action.get("tool", "unknown")
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

        return dict(sorted(tool_counts.items(), key=lambda x: x[1], reverse=True))
