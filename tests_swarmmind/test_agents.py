"""
SwarmMind Agent 测试
"""

import pytest
from unittest.mock import MagicMock, patch

from swarmmind.agents.base import BaseAgent
from swarmmind.agents.planner import PlannerAgent, PlanResult
from swarmmind.agents.executor import ExecutorAgent
from swarmmind.agents.reviewer import ReviewerAgent, ReviewResult
from swarmmind.core.security import AgentPermission


class TestBaseAgent:
    """Agent 基类测试"""

    def test_agent_permission(self):
        """测试 Agent 权限"""
        # 模拟 LLM
        with patch('swarmmind.agents.planner.get_provider') as mock_provider:
            mock_provider.return_value = MagicMock()

            agent = PlannerAgent()
            assert agent.permission == AgentPermission.READ_ONLY

    def test_validate_permission(self):
        """测试权限验证"""
        with patch('swarmmind.agents.planner.get_provider') as mock_provider:
            mock_provider.return_value = MagicMock()

            agent = PlannerAgent()

            # Planner 应该能访问 read_only 工具
            assert agent.validate_permission("read_workspace_file")
            assert agent.validate_permission("list_workspace_files")

            # Planner 不应该能访问写入工具
            assert not agent.validate_permission("write_workspace_file")
            assert not agent.validate_permission("execute_workspace_shell")


class TestPlannerAgent:
    """规划 Agent 测试"""

    def test_planner_initialization(self):
        """测试初始化"""
        with patch('swarmmind.agents.planner.get_provider') as mock_provider:
            mock_provider.return_value = MagicMock()

            agent = PlannerAgent(provider_name="openai", model_name="gpt-4")

            assert agent.name == "planner"
            assert agent.role == "任务规划者"
            assert agent.permission == AgentPermission.READ_ONLY

    def test_planner_tools_filtered(self):
        """测试工具过滤"""
        with patch('swarmmind.agents.planner.get_provider') as mock_provider:
            mock_provider.return_value = MagicMock()

            agent = PlannerAgent()

            # 应该只有 read_only 工具
            tool_names = [t.name for t in agent.tools]
            assert "read_workspace_file" in tool_names
            assert "write_workspace_file" not in tool_names


class TestExecutorAgent:
    """执行 Agent 测试"""

    def test_executor_initialization(self):
        """测试初始化"""
        with patch('swarmmind.agents.executor.get_provider') as mock_provider:
            mock_provider.return_value = MagicMock()

            agent = ExecutorAgent()

            assert agent.name == "executor"
            assert agent.role == "任务执行者"
            assert agent.permission == AgentPermission.STANDARD

    def test_executor_tools_includes_write(self):
        """测试执行 Agent 包含写入工具"""
        with patch('swarmmind.agents.executor.get_provider') as mock_provider:
            mock_provider.return_value = MagicMock()

            agent = ExecutorAgent()

            tool_names = [t.name for t in agent.tools]
            assert "write_workspace_file" in tool_names
            assert "execute_workspace_shell" in tool_names


class TestReviewerAgent:
    """审查 Agent 测试"""

    def test_reviewer_initialization(self):
        """测试初始化"""
        with patch('swarmmind.agents.reviewer.get_provider') as mock_provider:
            mock_provider.return_value = MagicMock()

            agent = ReviewerAgent()

            assert agent.name == "reviewer"
            assert agent.role == "结果审查者"
            assert agent.permission == AgentPermission.READ_ONLY
