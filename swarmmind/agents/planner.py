"""
SwarmMind Planner Agent

负责分析用户意图，制定执行计划。
权限等级: read_only
"""

from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseAgent
from ..core.security import AgentPermission
from ..core.provider import get_provider
from ..core.tools import BUILTIN_TOOLS


class PlanResult:
    """规划结果"""

    def __init__(self, analysis: str, actions: List[dict], reasoning: str):
        self.analysis = analysis
        self.actions = actions
        self.reasoning = reasoning

    def __repr__(self) -> str:
        return f"<PlanResult actions={len(self.actions)}>"


class PlannerAgent(BaseAgent):
    """
    规划 Agent
    - 分析用户意图
    - 制定执行计划
    - 权限等级: read_only
    """

    name = "planner"
    role = "任务规划者"
    permission = AgentPermission.READ_ONLY

    def __init__(self, provider_name: str = "openai", model_name: str = "gpt-4o-mini", experience_store=None):
        super().__init__(tools=BUILTIN_TOOLS)
        self.llm = get_provider(provider_name=provider_name, model_name=model_name)
        self.tools = [t for t in BUILTIN_TOOLS if t.name in [
            "list_workspace_files",
            "read_workspace_file",
            "get_current_time",
            "get_system_model_info"
        ]]
        self.experience_store = experience_store

    def _build_system_prompt(self, similar_experiences: str = "") -> str:
        experience_section = ""
        if similar_experiences:
            experience_section = f"""
【相似历史经验】
以下是与当前任务相似的历史执行经验，请参考：
{similar_experiences}
"""

        return f"""你是 SwarmMind 的规划 Agent，负责分析用户意图并制定执行计划。

{experience_section}
【核心职责】
1. 分析用户的真实需求
2. 对于内容创作任务，直接在 analysis 字段生成完整内容
3. 对于文件操作任务，制定具体的工具调用步骤
4. 评估任务风险等级

【可用工具列表】
- list_workspace_files: 列出 workspace/office 目录中的文件
- read_workspace_file: 读取文件内容
- write_workspace_file: 写入文件内容
- execute_workspace_shell: 执行 Shell 命令
- get_current_time: 获取当前时间
- get_system_model_info: 获取模型信息
- calculator: 数学计算

【重要规则】
1. 如果用户要求生成内容并保存，analysis 字段应包含完整生成的内容，actions 用于保存操作
2. 如果用户的请求不需要文件操作，analysis 字段直接回答用户问题
3. 不要编造不存在的工具名称
4. 如果有相似历史经验，优先参考成功的方案

【输出格式】
严格按以下 JSON 格式输出（不要包含markdown代码块标记）：

对于内容生成+保存任务：
{{"analysis": "完整生成的内容（用户要保存的内容）", "actions": [{{"tool": "write_workspace_file", "args": {{"filepath": "文件名", "content": "同analysis中的内容"}}]}}}}

对于纯对话任务：
{{"analysis": "直接回答用户的问题", "actions": [], "reasoning": "无需工具操作"}}

对于其他文件操作任务：
{{"analysis": "任务分析简述", "actions": [{{"tool": "工具名", "args": {{...}}, "reason": "原因"}}], "reasoning": "思路"}}

【安全协议】
- 所有文件操作限制在 workspace/office 目录内
- 禁止任何越权操作
- 高风险操作必须标记
"""

    async def run(self, user_input: str) -> PlanResult:
        """分析用户输入，生成执行计划"""
        # 检索相似经验
        similar_experiences = ""
        self._last_experience_info = ""
        if self.experience_store:
            try:
                experiences = await self.experience_store.recall(user_input, top_k=3, success_only=True)
                if experiences:
                    self._last_experience_info = f"Found {len(experiences)} similar experiences"
                    similar_experiences = "\n".join([
                        f"- 任务: {exp.task}\n  结果: {exp.result[:200]}...\n  成功: ✅"
                        for exp in experiences[:3]
                    ])
                else:
                    self._last_experience_info = "No similar experiences found"
            except Exception as e:
                self._last_experience_info = f"Recall failed: {e}"
        else:
            self._last_experience_info = "Experience store not enabled"

        messages = [
            SystemMessage(content=self._build_system_prompt(similar_experiences)),
            HumanMessage(content=f"请分析以下用户请求并制定执行计划：\n\n{user_input}")
        ]

        response = self.llm.invoke(messages)

        import json
        import re

        content = response.content.strip()

        # 去除 markdown 代码块标记
        if content.startswith("```"):
            # 去掉第一行 (```json 或 ```) 和最后一行 (```)
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        try:
            result = json.loads(content)
            return PlanResult(
                analysis=result.get("analysis", ""),
                actions=result.get("actions", []),
                reasoning=result.get("reasoning", "")
            )
        except json.JSONDecodeError:
            # 尝试从文本中提取 JSON（非贪婪匹配）
            json_match = re.search(r'\{[^\0]*?\}', content)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return PlanResult(
                        analysis=result.get("analysis", ""),
                        actions=result.get("actions", []),
                        reasoning=result.get("reasoning", "")
                    )
                except json.JSONDecodeError:
                    pass

            # 无法解析，返回原始响应
            return PlanResult(
                analysis=content,
                actions=[],
                reasoning=""
            )

    async def stream(self, user_input: str):
        """流式输出规划过程"""
        # 检索相似经验（流式模式暂不支持经验检索）
        messages = [
            SystemMessage(content=self._build_system_prompt("")),
            HumanMessage(content=f"请分析以下用户请求并制定执行计划：\n\n{user_input}")
        ]

        async for chunk in self.llm.astream(messages):
            yield chunk.content
