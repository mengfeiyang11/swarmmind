"""
SwarmMind Reviewer Agent

负责审查执行结果，确保质量。
权限等级: read_only
"""

from typing import Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseAgent
from ..core.security import AgentPermission
from ..core.provider import get_provider
from ..core.tools import BUILTIN_TOOLS


class ReviewResult:
    """审查结果"""

    def __init__(self, passed: bool, feedback: str, suggestions: list):
        self.passed = passed
        self.feedback = feedback
        self.suggestions = suggestions

    def __repr__(self) -> str:
        status = "✅ 通过" if self.passed else "❌ 需改进"
        return f"<ReviewResult {status} feedback='{self.feedback[:50]}...'>"


class ReviewerAgent(BaseAgent):
    """
    审查 Agent
    - 检查执行结果
    - 确保任务完成质量
    - 权限等级: read_only
    """

    name = "reviewer"
    role = "结果审查者"
    permission = AgentPermission.READ_ONLY

    def __init__(self, provider_name: str = "openai", model_name: str = "gpt-4o-mini"):
        super().__init__(tools=BUILTIN_TOOLS)
        self.llm = get_provider(provider_name=provider_name, model_name=model_name)
        self.tools = [t for t in BUILTIN_TOOLS if t.name in [
            "list_workspace_files",
            "read_workspace_file",
            "get_current_time"
        ]]

    def _build_system_prompt(self) -> str:
        return """你是 SwarmMind 的审查 Agent，负责检查执行结果。

【核心职责】
1. 验证任务是否正确完成
2. 检查是否有遗漏或错误
3. 提供改进建议

【审查标准】
- 功能是否正确实现
- 代码是否符合规范
- 是否存在安全隐患
- 是否有优化空间

【输出格式】
{
    "passed": true/false,
    "feedback": "审查反馈",
    "suggestions": ["建议1", "建议2", ...]
}
"""

    async def run(self, execution_result: str) -> ReviewResult:
        """审查执行结果"""
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=f"请审查以下执行结果：\n\n{execution_result}")
        ]

        response = self.llm.invoke(messages)

        import json
        try:
            result = json.loads(response.content)
            return ReviewResult(
                passed=result.get("passed", False),
                feedback=result.get("feedback", ""),
                suggestions=result.get("suggestions", [])
            )
        except json.JSONDecodeError:
            return ReviewResult(
                passed=True,
                feedback=response.content,
                suggestions=[]
            )

    async def stream(self, execution_result: str):
        """流式输出审查过程"""
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=f"请审查以下执行结果：\n\n{execution_result}")
        ]

        async for chunk in self.llm.astream(messages):
            yield chunk.content
