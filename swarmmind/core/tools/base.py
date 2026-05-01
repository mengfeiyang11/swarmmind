"""
SwarmMind 工具基类
"""

from typing import Any, Type
from langchain_core.tools import BaseTool, tool
from abc import ABC, abstractmethod
import asyncio
from pydantic import BaseModel, Field


# 函数装饰器模式（简单工具）
swarmmind_tool = tool


# 类模式工具（复杂场景）
class SwarmMindBaseTool(BaseTool, ABC):
    """
    SwarmMind 工具基类。
    适合需要复杂初始化逻辑或需要保存内部状态的工具。
    """

    name: str
    description: str
    args_schema: Type[BaseModel]

    # 权限等级
    required_permission: str = "standard"

    # 风险等级
    risk_level: str = "low"

    @abstractmethod
    def _run(self, **kwargs: Any) -> Any:
        """工具的同步执行逻辑，子类必须实现。"""
        raise NotImplementedError("子类必须实现 _run 方法")

    async def _arun(self, **kwargs: Any) -> Any:
        """工具的异步执行逻辑（可选）。"""
        return await asyncio.to_thread(self._run, **kwargs)