"""
SwarmMind Agent 基类

提供:
- 权限管理
- 工具绑定
- 安全检查
"""

from typing import List, Optional, Any
from abc import ABC, abstractmethod
from langchain_core.tools import BaseTool

from ..core.security import (
    AgentPermission,
    SafetyChecker,
    ConfirmProtocol,
    RiskLevel
)
from ..core.tools import BUILTIN_TOOLS


class BaseAgent(ABC):
    """
    SwarmMind Agent 基类
    """

    name: str = "base_agent"
    role: str = "基础 Agent"
    permission: AgentPermission = AgentPermission.READ_ONLY

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        permission: Optional[AgentPermission] = None
    ):
        if permission:
            self.permission = permission

        # 根据权限过滤工具
        self.tools = self._filter_tools_by_permission(tools or BUILTIN_TOOLS)

    def _filter_tools_by_permission(self, tools: List[BaseTool]) -> List[BaseTool]:
        """根据权限过滤工具"""
        allowed_tools = SafetyChecker.get_agent_tools(self.permission)

        if "*" in allowed_tools:
            return tools

        return [t for t in tools if t.name in allowed_tools]

    def validate_permission(self, tool_name: str) -> bool:
        """检查是否有权限调用该工具"""
        return SafetyChecker.check_permission(self.permission, tool_name)

    def get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    @abstractmethod
    async def run(self, input_data: Any) -> Any:
        """执行 Agent（子类实现）"""
        pass

    @abstractmethod
    async def stream(self, input_data: Any):
        """流式执行 Agent（子类实现）"""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} permission={self.permission.value}>"
