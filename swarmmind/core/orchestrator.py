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
import time
from typing import AsyncIterator, Optional, Any, List, Dict
from rich.console import Console
from rich.panel import Panel

from .memory import MemorySystem
from .logger import audit_logger
from .security import SafetyChecker, ConfirmProtocol
from .config import MEMORY_DIR
from .experience import ExperienceStore
from .anomaly import BehaviorMonitor, AnomalySeverity, RecommendedAction

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

    async def run(self, user_input: str) -> str:
        """完整执行流程"""
        start_time = time.time()
        console.print("\n[bold cyan]=== SwarmMind Multi-Agent ====[/bold cyan]\n")

        # Step 1: Planner 分析
        console.print("[bold yellow][Planner] Analyzing task...[/bold yellow]")
        plan = await self.planner.run(user_input)

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
            return plan.analysis

        # Step 3: 安全检查
        console.print("[bold red][Safety] Checking permissions...[/bold red]")
        confirmed_actions = await self._safety_check(plan)

        if not confirmed_actions:
            console.print("[bold red]Task blocked by security protocol[/bold red]")
            return "Task blocked by security protocol"

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
                    return f"Task blocked by anomaly detection: {anomaly_report.description}"

        # Step 5: Executor 执行（支持并行）
        console.print("[bold green][Executor] Executing task...[/bold green]")

        # 检查是否有并行组
        parallel_groups = getattr(plan, 'parallel_groups', None) or confirmed_actions.get("parallel_groups", [])

        if self.enable_parallel and parallel_groups:
            result = await self._execute_parallel(confirmed_actions, parallel_groups)
        else:
            result = await self.executor.run(confirmed_actions)

        console.print(Panel(
            str(result)[:500],
            title="Execution Result",
            border_style="green"
        ))

        # Step 6: Reviewer 审查
        console.print("[bold blue][Reviewer] Checking result...[/bold blue]")
        review = await self.reviewer.run(str(result))

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

        # Step 1: Planner 分析（包含向量检索）
        yield "\n[Planner] Analyzing...\n\n"
        plan = await self.planner.run(user_input)

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
                yield "\nTask blocked by security protocol\n"
                return

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
