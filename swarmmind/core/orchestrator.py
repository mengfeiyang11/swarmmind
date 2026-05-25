"""
SwarmMind 多 Agent 编排器

安全的多 Agent 协作编排：
1. Planner 分析意图
2. SafetyChecker 检查风险
3. Executor 执行任务（支持并行）
4. Reviewer 审查结果
5. 经验回放记录
6. 异常行为检测
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import AsyncIterator, Optional, Any, List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from rich.console import Console
from rich.panel import Panel

from .memory import MemorySystem
from .logger import audit_logger
from .security import SafetyChecker, ConfirmProtocol
from .config import MEMORY_DIR
from .experience import ExperienceStore
from .anomaly import BehaviorMonitor, AnomalySeverity, RecommendedAction
from .compressor import ContextCompressor

console = Console()


class SafeOrchestrator:
    """
    安全的多 Agent 编排器

    执行流程:
    1. Planner 分析用户意图，制定计划
    2. SafetyChecker 检查每个步骤的风险
    3. 高风险操作请求用户确认
    4. Executor 执行已确认的计划（支持并行）
    5. Reviewer 检查执行结果
    6. 记录执行经验
    """

    def __init__(
        self,
        provider_name: str = "openai",
        model_name: str = "gpt-4o-mini",
        enable_parallel: bool = True,
        enable_experience: bool = True,
        enable_anomaly_detection: bool = True
    ):
        self.provider_name = provider_name
        self.model_name = model_name
        self.enable_parallel = enable_parallel
        self.enable_experience = enable_experience
        self.enable_anomaly_detection = enable_anomaly_detection

        self.memory = MemorySystem(MEMORY_DIR)

        # 经验回放系统
        if enable_experience:
            self.experience_store = ExperienceStore(MEMORY_DIR)
        else:
            self.experience_store = None

        # 异常行为检测
        if enable_anomaly_detection:
            self.behavior_monitor = BehaviorMonitor()
        else:
            self.behavior_monitor = None

        # 延迟导入以避免循环依赖
        from ..agents.planner import PlannerAgent
        from ..agents.executor import ExecutorAgent
        from ..agents.reviewer import ReviewerAgent

        self.planner = PlannerAgent(provider_name, model_name, self.experience_store)
        self.executor = ExecutorAgent(provider_name, model_name)
        self.reviewer = ReviewerAgent(provider_name, model_name)

        self.summary = ""
        self._conversation_messages: List[BaseMessage] = []
        self._compressor = ContextCompressor(self.planner.llm)
        self._checkpoint_path = os.path.join(MEMORY_DIR, "checkpoint.json")

    async def run(self, user_input: str) -> str:
        """完整执行流程"""
        start_time = time.time()
        console.print("\n[bold cyan]=== SwarmMind Multi-Agent ====[/bold cyan]\n")
        checkpoint = self._init_checkpoint(user_input)

        # Step 1: Planner 分析
        console.print("[bold yellow][Planner] Analyzing task...[/bold yellow]")
        plan = await self.planner.run(user_input, context_summary=self.summary)
        self._update_checkpoint(checkpoint, stage="planned", plan=self._plan_to_dict(plan))

        console.print(Panel(
            f"[bold]Analysis:[/bold] {plan.analysis[:300]}...\n"
            f"[bold]Steps:[/bold] {len(plan.actions)}\n"
            f"[bold]Reasoning:[/bold] {plan.reasoning[:200]}",
            title="Plan Result",
            border_style="yellow"
        ))

        # Step 2: 检查是否是纯对话（无需工具）
        if not plan.actions or all(a.get("tool") in ["direct_response", "none", ""] for a in plan.actions):
            # 直接返回分析结果
            console.print("[bold green][Info] Direct response - no tool execution needed[/bold green]")
            self._record_interaction(user_input, plan.analysis)
            self._finalize_checkpoint(checkpoint, plan.analysis)
            return plan.analysis

        # Step 3: 安全检查
        console.print("[bold red][Safety] Checking permissions...[/bold red]")
        confirmed_actions = await self._safety_check(plan)

        if not confirmed_actions:
            console.print("[bold red]Task blocked by security protocol[/bold red]")
            blocked_message = "Task blocked by security protocol"
            self._record_interaction(user_input, blocked_message)
            self._finalize_checkpoint(checkpoint, blocked_message)
            return blocked_message
        self._update_checkpoint(checkpoint, stage="confirmed", confirmed_actions=confirmed_actions)

        # Step 4: 异常行为检测
        if self.enable_anomaly_detection and self.behavior_monitor:
            for action in confirmed_actions.get("actions", []):
                anomaly_report = self.behavior_monitor.check_anomaly(
                    agent="executor",
                    tool=action.get("tool", ""),
                    args=action.get("args", {})
                )
                if anomaly_report.is_anomaly and anomaly_report.recommended_action == RecommendedAction.BLOCK:
                    console.print(f"[bold red]Anomaly detected: {anomaly_report.description}[/bold red]")
                    blocked_message = f"Task blocked by anomaly detection: {anomaly_report.description}"
                    self._record_interaction(user_input, blocked_message)
                    self._finalize_checkpoint(checkpoint, blocked_message)
                    return blocked_message

        # Step 5: Executor 执行（支持并行）
        console.print("[bold green][Executor] Executing task...[/bold green]")

        # 检查是否有并行组
        parallel_groups = getattr(plan, 'parallel_groups', None) or confirmed_actions.get("parallel_groups", [])

        if self.enable_parallel and parallel_groups:
            result = await self._execute_parallel(confirmed_actions, parallel_groups)
        else:
            result = await self.executor.run(confirmed_actions)
        self._update_checkpoint(checkpoint, stage="executed", result=str(result))

        console.print(Panel(
            str(result)[:500],
            title="Execution Result",
            border_style="green"
        ))

        # Step 6: Reviewer 审查
        console.print("[bold blue][Reviewer] Checking result...[/bold blue]")
        review = await self.reviewer.run(str(result))
        self._update_checkpoint(
            checkpoint,
            stage="reviewed",
            review={"passed": review.passed, "feedback": review.feedback, "suggestions": review.suggestions}
        )

        status = "PASS" if review.passed else "NEEDS IMPROVEMENT"
        console.print(Panel(
            f"[bold]Review:[/bold] {status}\n"
            f"[bold]Feedback:[/bold] {review.feedback}\n"
            f"[bold]Suggestions:[/bold] {', '.join(review.suggestions) if review.suggestions else 'None'}",
            title="Review Result",
            border_style="blue"
        ))

        # Step 7: 记录执行经验
        duration = time.time() - start_time
        if self.enable_experience and self.experience_store:
            await self.experience_store.record(
                task=user_input,
                plan={"analysis": plan.analysis, "actions": confirmed_actions.get("actions", [])},
                result=str(result),
                review_passed=review.passed,
                review_feedback=review.feedback,
                suggestions=review.suggestions,
                duration_seconds=duration
            )
            console.print(f"[dim][Experience] Recorded (duration: {duration:.2f}s)[/dim]")

        # 记录审计日志
        audit_logger.log_event(
            thread_id="orchestrator",
            event="task_complete",
            agent_name="orchestrator",
            risk_level="low",
            plan_summary=plan.analysis[:200],
            review_passed=review.passed
        )

        self._record_interaction(user_input, str(result))
        self._finalize_checkpoint(checkpoint, str(result))
        return result

    async def _execute_parallel(self, plan: dict, parallel_groups: List[List[Dict]]) -> str:
        """
        并行执行任务

        参数:
        - plan: 执行计划
        - parallel_groups: 并行步骤组列表，每组内的步骤可以并行执行

        返回:
        - 执行结果汇总
        """
        results = []
        sequential_actions = plan.get("actions", [])

        # 先执行顺序步骤
        if sequential_actions and not parallel_groups:
            result = await self.executor.run(plan)
            results.append(str(result))
            return "\n\n".join(results) if results else ""

        # 对每个并行组进行并行执行
        for group_idx, group in enumerate(parallel_groups):
            console.print(f"[bold green][Executor] Running parallel group {group_idx + 1} ({len(group)} tasks)...[/bold green]")

            # 创建并行任务
            tasks = []
            for action in group:
                single_plan = {
                    "analysis": f"Execute {action.get('tool', 'unknown')}",
                    "actions": [action],
                    "reasoning": "Parallel execution"
                }
                tasks.append(self.executor.run(single_plan))

            # 并行执行
            if tasks:
                group_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, gr in enumerate(group_results):
                    if isinstance(gr, Exception):
                        results.append(f"Task {i + 1} failed: {str(gr)}")
                    else:
                        results.append(str(gr))

                    # 记录行为
                    if self.behavior_monitor:
                        self.behavior_monitor.record_action(
                            agent="executor",
                            tool=group[i].get("tool", "unknown"),
                            args=group[i].get("args", {}),
                            result=str(gr)[:200]
                        )

        return "\n\n".join(results) if results else "No results from parallel execution"

    async def stream(self, user_input: str) -> AsyncIterator[str]:
        """流式执行流程"""
        start_time = time.time()
        checkpoint = self._init_checkpoint(user_input)

        # Step 1: Planner 分析（包含向量检索）
        yield "\n[Planner] Analyzing...\n\n"
        plan = await self.planner.run(user_input, context_summary=self.summary)
        self._update_checkpoint(checkpoint, stage="planned", plan=self._plan_to_dict(plan))

        # 输出经验检索状态（通过 planner 实例获取）
        exp_info = getattr(self.planner, '_last_experience_info', '')
        if exp_info:
            yield f"[dim cyan][Experience] {exp_info}[/dim cyan]\n\n"

        # 直接输出分析结果
        yield plan.analysis

        # 如果有需要执行的工具操作
        if plan.actions:
            yield "\n\n[Safety] Checking permissions...\n\n"
            confirmed = await self._safety_check(plan)

            if not confirmed:
                blocked_message = "Task blocked by security protocol"
                yield f"\n{blocked_message}\n"
                self._record_interaction(user_input, blocked_message)
                self._finalize_checkpoint(checkpoint, blocked_message)
                return
            self._update_checkpoint(checkpoint, stage="confirmed", confirmed_actions=confirmed)

            yield "[Executor] Running...\n\n"
            result_text = ""
            first_chunk = True
            async for chunk in self.executor.stream(confirmed):
                if first_chunk:
                    # 跳过 executor 第一条重复消息（executor 第一条总是对 plan 的响应）
                    first_chunk = False
                    continue
                result_text += chunk
                yield chunk

            yield "\n\n[Reviewer] Checking...\n\n"
            review = await self.reviewer.run(result_text)
            self._update_checkpoint(
                checkpoint,
                stage="reviewed",
                review={"passed": review.passed, "feedback": review.feedback, "suggestions": review.suggestions},
                result=result_text
            )

            if review.passed:
                yield "\nReview: PASS"
            else:
                yield f"\nReview: NEEDS IMPROVEMENT - {review.feedback}"

            # 记录经验（流式模式）
            duration = time.time() - start_time
            if self.enable_experience and self.experience_store:
                await self.experience_store.record(
                    task=user_input,
                    plan={"analysis": plan.analysis, "actions": confirmed.get("actions", [])},
                    result=result_text,
                    review_passed=review.passed,
                    review_feedback=review.feedback,
                    suggestions=review.suggestions,
                    duration_seconds=duration
                )
                yield f"\n\n[dim][Experience] Recorded (duration: {duration:.2f}s)[/dim]"
            self._record_interaction(user_input, result_text)
            self._finalize_checkpoint(checkpoint, result_text)
        else:
            self._record_interaction(user_input, plan.analysis)
            self._finalize_checkpoint(checkpoint, plan.analysis)

    async def resume(self) -> str:
        """从断点续跑"""
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            return "No checkpoint found"
        return await self._resume_from_checkpoint(checkpoint)

    async def stream_resume(self) -> AsyncIterator[str]:
        """流式断点续跑"""
        checkpoint = self._load_checkpoint()
        if not checkpoint:
            yield "No checkpoint found"
            return
        async for chunk in self._stream_from_checkpoint(checkpoint):
            yield chunk

    async def _safety_check(self, plan: Any) -> Optional[dict]:
        """
        安全检查
        - 检查每个步骤的风险等级
        - 高风险操作请求用户确认
        """
        confirmed_actions = []

        for action in plan.actions:
            tool_name = action.get("tool", "")

            # 跳过无需工具的操作
            if tool_name in ["direct_response", "none", ""]:
                continue

            risk_level = ConfirmProtocol.get_risk_level(tool_name)

            # 权限检查
            if not SafetyChecker.check_permission(self.executor.permission, tool_name):
                console.print(f"[red]Permission denied: {tool_name}[/red]")
                audit_logger.log_event(
                    thread_id="orchestrator",
                    event="permission_denied",
                    agent_name="executor",
                    risk_level=risk_level.value,
                    tool=tool_name
                )
                continue

            # 高风险操作确认
            if ConfirmProtocol.require_confirmation(tool_name, action.get("args", {})):
                console.print(f"[bold red]High-risk action: {tool_name} (risk: {risk_level.value})[/bold red]")

                confirmed = self._request_user_confirmation(tool_name, action)
                if not confirmed:
                    console.print(f"[red]User rejected: {tool_name}[/red]")
                    audit_logger.log_event(
                        thread_id="orchestrator",
                        event="user_rejected",
                        agent_name="executor",
                        risk_level=risk_level.value,
                        tool=tool_name,
                        user_confirmed=False
                    )
                    continue

                audit_logger.log_event(
                    thread_id="orchestrator",
                    event="user_confirmed",
                    agent_name="executor",
                    risk_level=risk_level.value,
                    tool=tool_name,
                    user_confirmed=True
                )

            confirmed_actions.append(action)

        # 如果没有需要执行的工具操作，返回 None
        if not confirmed_actions:
            return None

        return {
            "analysis": plan.analysis,
            "actions": confirmed_actions,
            "reasoning": plan.reasoning
        }

    def _request_user_confirmation(self, tool_name: str, action: dict) -> bool:
        """请求用户确认危险操作"""
        console.print(Panel(
            f"[bold red]High-risk action warning[/bold red]\n\n"
            f"Tool: {tool_name}\n"
            f"Args: {action.get('args', {})}\n"
            f"Reason: {action.get('reason', 'High-risk operation')}\n\n"
            f"Allow execution?",
            title="Security Confirmation",
            border_style="red"
        ))

        try:
            response = input("[y/N] ").strip().lower()
            return response in ["y", "yes"]
        except (EOFError, KeyboardInterrupt):
            return False

    def _init_checkpoint(self, user_input: str) -> dict:
        checkpoint = {
            "stage": "start",
            "user_input": user_input,
            "summary": self.summary,
            "plan": None,
            "confirmed_actions": None,
            "result": None,
            "review": None,
            "created_at": datetime.now().isoformat()
        }
        self._save_checkpoint(checkpoint)
        return checkpoint

    def _plan_to_dict(self, plan: Any) -> Dict[str, Any]:
        data = {
            "analysis": getattr(plan, "analysis", ""),
            "actions": getattr(plan, "actions", []),
            "reasoning": getattr(plan, "reasoning", "")
        }
        parallel_groups = getattr(plan, "parallel_groups", None)
        if parallel_groups:
            data["parallel_groups"] = parallel_groups
        return data

    def _build_plan(self, data: Dict[str, Any]):
        from ..agents.planner import PlanResult

        plan = PlanResult(
            analysis=data.get("analysis", ""),
            actions=data.get("actions", []),
            reasoning=data.get("reasoning", "")
        )
        if "parallel_groups" in data:
            setattr(plan, "parallel_groups", data.get("parallel_groups"))
        return plan

    def _save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["summary"] = self.summary
        checkpoint["updated_at"] = datetime.now().isoformat()
        os.makedirs(os.path.dirname(self._checkpoint_path), exist_ok=True)
        with open(self._checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self._checkpoint_path):
            return None
        try:
            with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[Checkpoint] Load error: {e}")
            return None

    def _clear_checkpoint(self) -> None:
        if os.path.exists(self._checkpoint_path):
            try:
                os.remove(self._checkpoint_path)
            except Exception:
                pass

    def _update_checkpoint(self, checkpoint: Dict[str, Any], **updates) -> None:
        checkpoint.update(updates)
        self._save_checkpoint(checkpoint)

    def _finalize_checkpoint(self, checkpoint: Dict[str, Any], result: str) -> None:
        checkpoint.update({"stage": "completed", "result": result})
        self._save_checkpoint(checkpoint)
        self._clear_checkpoint()

    def _check_anomaly_block(self, confirmed_actions: Dict[str, Any]) -> Optional[str]:
        if not (self.enable_anomaly_detection and self.behavior_monitor):
            return None

        for action in confirmed_actions.get("actions", []):
            anomaly_report = self.behavior_monitor.check_anomaly(
                agent="executor",
                tool=action.get("tool", ""),
                args=action.get("args", {})
            )
            if anomaly_report.is_anomaly and anomaly_report.recommended_action == RecommendedAction.BLOCK:
                return f"Task blocked by anomaly detection: {anomaly_report.description}"

        return None

    async def _resume_from_checkpoint(self, checkpoint: Dict[str, Any]) -> str:
        stage = checkpoint.get("stage", "start")
        user_input = checkpoint.get("user_input", "")
        self.summary = checkpoint.get("summary", "")

        if stage == "start":
            return await self.run(user_input)

        plan_data = checkpoint.get("plan") or {}
        plan = self._build_plan(plan_data)

        if not plan.actions or all(a.get("tool") in ["direct_response", "none", ""] for a in plan.actions):
            self._record_interaction(user_input, plan.analysis)
            self._finalize_checkpoint(checkpoint, plan.analysis)
            return plan.analysis

        confirmed_actions = checkpoint.get("confirmed_actions")
        if stage in ["planned"] or not confirmed_actions:
            confirmed_actions = await self._safety_check(plan)
            if not confirmed_actions:
                blocked_message = "Task blocked by security protocol"
                self._record_interaction(user_input, blocked_message)
                self._finalize_checkpoint(checkpoint, blocked_message)
                return blocked_message
            self._update_checkpoint(checkpoint, stage="confirmed", confirmed_actions=confirmed_actions)

        anomaly_block = self._check_anomaly_block(confirmed_actions)
        if anomaly_block:
            self._record_interaction(user_input, anomaly_block)
            self._finalize_checkpoint(checkpoint, anomaly_block)
            return anomaly_block

        result = checkpoint.get("result")
        if stage in ["confirmed"] or result is None:
            parallel_groups = plan_data.get("parallel_groups") or confirmed_actions.get("parallel_groups", [])
            if self.enable_parallel and parallel_groups:
                result = await self._execute_parallel(confirmed_actions, parallel_groups)
            else:
                result = await self.executor.run(confirmed_actions)
            self._update_checkpoint(checkpoint, stage="executed", result=str(result))

        review_data = checkpoint.get("review")
        if stage in ["executed"] or not review_data:
            review = await self.reviewer.run(str(result))
            review_data = {
                "passed": review.passed,
                "feedback": review.feedback,
                "suggestions": review.suggestions
            }
            self._update_checkpoint(checkpoint, stage="reviewed", review=review_data)

        duration = 0.0
        if self.enable_experience and self.experience_store:
            await self.experience_store.record(
                task=user_input,
                plan={"analysis": plan.analysis, "actions": confirmed_actions.get("actions", [])},
                result=str(result),
                review_passed=review_data.get("passed", False),
                review_feedback=review_data.get("feedback", ""),
                suggestions=review_data.get("suggestions", []),
                duration_seconds=duration
            )

        self._record_interaction(user_input, str(result))
        self._finalize_checkpoint(checkpoint, str(result))
        return str(result)

    async def _stream_from_checkpoint(self, checkpoint: Dict[str, Any]) -> AsyncIterator[str]:
        stage = checkpoint.get("stage", "start")
        user_input = checkpoint.get("user_input", "")
        self.summary = checkpoint.get("summary", "")

        if stage == "start":
            async for chunk in self.stream(user_input):
                yield chunk
            return

        plan_data = checkpoint.get("plan") or {}
        plan = self._build_plan(plan_data)

        yield "\n[Planner] Resuming...\n\n"
        yield plan.analysis

        if not plan.actions or all(a.get("tool") in ["direct_response", "none", ""] for a in plan.actions):
            self._record_interaction(user_input, plan.analysis)
            self._finalize_checkpoint(checkpoint, plan.analysis)
            return

        yield "\n\n[Safety] Checking permissions...\n\n"
        confirmed_actions = checkpoint.get("confirmed_actions")
        if stage in ["planned"] or not confirmed_actions:
            confirmed_actions = await self._safety_check(plan)
            if not confirmed_actions:
                blocked_message = "Task blocked by security protocol"
                yield f"\n{blocked_message}\n"
                self._record_interaction(user_input, blocked_message)
                self._finalize_checkpoint(checkpoint, blocked_message)
                return
            self._update_checkpoint(checkpoint, stage="confirmed", confirmed_actions=confirmed_actions)

        anomaly_block = self._check_anomaly_block(confirmed_actions)
        if anomaly_block:
            yield f"\n{anomaly_block}\n"
            self._record_interaction(user_input, anomaly_block)
            self._finalize_checkpoint(checkpoint, anomaly_block)
            return

        result_text = checkpoint.get("result", "")
        if stage in ["confirmed"] or not result_text:
            yield "[Executor] Running...\n\n"
            first_chunk = True
            async for chunk in self.executor.stream(confirmed_actions):
                if first_chunk:
                    first_chunk = False
                    continue
                result_text += chunk
                yield chunk
            self._update_checkpoint(checkpoint, stage="executed", result=result_text)
        else:
            yield "\n[Executor] Resumed previous result\n\n"
            yield result_text

        yield "\n\n[Reviewer] Checking...\n\n"
        review_data = checkpoint.get("review")
        if stage in ["executed"] or not review_data:
            review = await self.reviewer.run(result_text)
            review_data = {
                "passed": review.passed,
                "feedback": review.feedback,
                "suggestions": review.suggestions
            }
            self._update_checkpoint(checkpoint, stage="reviewed", review=review_data)

        if review_data.get("passed"):
            yield "\nReview: PASS"
        else:
            yield f"\nReview: NEEDS IMPROVEMENT - {review_data.get('feedback', '')}"

        if self.enable_experience and self.experience_store:
            await self.experience_store.record(
                task=user_input,
                plan={"analysis": plan.analysis, "actions": confirmed_actions.get("actions", [])},
                result=result_text,
                review_passed=review_data.get("passed", False),
                review_feedback=review_data.get("feedback", ""),
                suggestions=review_data.get("suggestions", []),
                duration_seconds=0.0
            )

        self._record_interaction(user_input, result_text)
        self._finalize_checkpoint(checkpoint, result_text)

    def _record_interaction(self, user_input: str, result_text: str) -> None:
        """记录对话并在需要时更新摘要"""
        if not result_text:
            return

        self._conversation_messages.append(HumanMessage(content=user_input))
        self._conversation_messages.append(AIMessage(content=result_text))

        if not self._compressor.needs_compression(self._conversation_messages):
            return

        compressed = self._compressor.compress_sync(
            self._conversation_messages,
            existing_summary=self.summary
        )
        summary_text = self._extract_summary(compressed)
        if summary_text:
            self.summary = summary_text

        self._conversation_messages = [
            msg for msg in compressed if not isinstance(msg, SystemMessage)
        ]

    @staticmethod
    def _extract_summary(messages: List[BaseMessage]) -> str:
        for msg in messages:
            if (
                isinstance(msg, SystemMessage)
                and isinstance(msg.content, str)
                and msg.content.startswith("[历史对话摘要]")
            ):
                return msg.content.replace("[历史对话摘要]\n", "").strip()
        return ""
