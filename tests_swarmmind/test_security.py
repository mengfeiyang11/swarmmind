"""
SwarmMind 安全模块测试
"""

import pytest
from swarmmind.core.security import (
    AgentPermission,
    SafetyChecker,
    ConfirmProtocol,
    RiskLevel,
    HIGH_RISK_TOOLS,
    MEDIUM_RISK_TOOLS
)


class TestAgentPermission:
    """权限等级测试"""

    def test_permission_values(self):
        """测试权限枚举值"""
        assert AgentPermission.READ_ONLY.value == "read_only"
        assert AgentPermission.STANDARD.value == "standard"
        assert AgentPermission.ADMIN.value == "admin"


class TestSafetyChecker:
    """安全检查器测试"""

    def test_check_permission_read_only(self):
        """测试 read_only 权限"""
        # 允许的工具
        assert SafetyChecker.check_permission(AgentPermission.READ_ONLY, "list_workspace_files")
        assert SafetyChecker.check_permission(AgentPermission.READ_ONLY, "read_workspace_file")
        assert SafetyChecker.check_permission(AgentPermission.READ_ONLY, "get_current_time")

        # 禁止的工具
        assert not SafetyChecker.check_permission(AgentPermission.READ_ONLY, "write_workspace_file")
        assert not SafetyChecker.check_permission(AgentPermission.READ_ONLY, "execute_workspace_shell")

    def test_check_permission_standard(self):
        """测试 standard 权限"""
        # 允许的工具
        assert SafetyChecker.check_permission(AgentPermission.STANDARD, "read_workspace_file")
        assert SafetyChecker.check_permission(AgentPermission.STANDARD, "write_workspace_file")
        assert SafetyChecker.check_permission(AgentPermission.STANDARD, "execute_workspace_shell")

    def test_check_permission_admin(self):
        """测试 admin 权限"""
        # Admin 拥有所有权限
        assert SafetyChecker.check_permission(AgentPermission.ADMIN, "any_tool")
        assert SafetyChecker.check_permission(AgentPermission.ADMIN, "execute_workspace_shell")

    def test_is_high_risk(self):
        """测试高风险工具判断"""
        assert SafetyChecker.is_high_risk("execute_workspace_shell")
        assert not SafetyChecker.is_high_risk("get_current_time")

    def test_validate_command_safety(self):
        """测试命令安全性验证"""
        # 安全命令
        is_safe, _ = SafetyChecker.validate_command_safety("ls -la")
        assert is_safe

        # 危险命令 - 相对路径越权
        is_safe, msg = SafetyChecker.validate_command_safety("cat ../../../etc/passwd")
        assert not is_safe
        assert "相对路径越权" in msg

        # 危险命令 - Unix 绝对路径
        is_safe, msg = SafetyChecker.validate_command_safety("cat /etc/passwd")
        assert not is_safe

        # 危险命令 - Windows 盘符
        is_safe, msg = SafetyChecker.validate_command_safety("type C:\\Windows\\System32\\config\\SAM")
        assert not is_safe

    def test_get_agent_tools(self):
        """测试获取 Agent 工具列表"""
        read_only_tools = SafetyChecker.get_agent_tools(AgentPermission.READ_ONLY)
        assert "list_workspace_files" in read_only_tools
        assert "write_workspace_file" not in read_only_tools

        standard_tools = SafetyChecker.get_agent_tools(AgentPermission.STANDARD)
        assert "write_workspace_file" in standard_tools
        assert "execute_workspace_shell" in standard_tools


class TestConfirmProtocol:
    """确认协议测试"""

    def test_require_confirmation_high_risk(self):
        """测试高风险工具需要确认"""
        assert ConfirmProtocol.require_confirmation("execute_workspace_shell")
        assert ConfirmProtocol.require_confirmation("delete_file")

    def test_require_confirmation_low_risk(self):
        """测试低风险工具不需要确认"""
        assert not ConfirmProtocol.require_confirmation("get_current_time")
        assert not ConfirmProtocol.require_confirmation("list_workspace_files")

    def test_require_confirmation_batch_operation(self):
        """测试批量操作需要确认"""
        assert ConfirmProtocol.require_confirmation("write_workspace_file", {"batch_operation": True})

    def test_require_confirmation_write_mode(self):
        """测试覆盖模式写入需要确认"""
        assert ConfirmProtocol.require_confirmation("write_workspace_file", {"mode": "w"})
        assert not ConfirmProtocol.require_confirmation("write_workspace_file", {"mode": "a"})

    def test_get_risk_level(self):
        """测试获取风险等级"""
        assert ConfirmProtocol.get_risk_level("execute_workspace_shell") == RiskLevel.HIGH
        assert ConfirmProtocol.get_risk_level("write_workspace_file") == RiskLevel.MEDIUM
        assert ConfirmProtocol.get_risk_level("get_current_time") == RiskLevel.LOW
