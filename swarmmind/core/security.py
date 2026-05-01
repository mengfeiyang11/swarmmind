"""
SwarmMind 安全模块

提供:
- Agent 权限分级
- 危险操作确认协议
- 安全检查器
"""

from typing import Dict, List, Optional
from enum import Enum


# 高风险工具列表
HIGH_RISK_TOOLS = [
    "execute_workspace_shell",
    "execute_code",
    "run_python_script",
    "modify_scheduled_task",
    "delete_scheduled_task",
]

# 中风险工具列表
MEDIUM_RISK_TOOLS = [
    "write_workspace_file",
    "schedule_task",
]


class AgentPermission(str, Enum):
    """Agent 权限等级"""
    READ_ONLY = "read_only"    # 只能读取，不能写入
    STANDARD = "standard"      # 可读写，但危险命令需确认
    ADMIN = "admin"            # 完全权限


class RiskLevel(str, Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# 权限对应的允许工具
PERMISSION_TOOLS: Dict[AgentPermission, List[str]] = {
    AgentPermission.READ_ONLY: [
        "list_workspace_files",
        "read_workspace_file",
        "get_current_time",
        "get_system_model_info",
    ],
    AgentPermission.STANDARD: [
        "list_workspace_files",
        "read_workspace_file",
        "write_workspace_file",
        "execute_workspace_shell",
        "get_current_time",
        "get_system_model_info",
        "schedule_task",
        "list_scheduled_tasks",
    ],
    AgentPermission.ADMIN: [
        # Admin 拥有所有工具权限
        "*"
    ]
}


class ConfirmProtocol:
    """危险操作确认协议"""

    @staticmethod
    def require_confirmation(tool_name: str, args: dict = None) -> bool:
        """
        判断是否需要用户确认
        """
        if tool_name in HIGH_RISK_TOOLS:
            return True

        # 批量操作需要确认
        if args and args.get("batch_operation"):
            return True

        # 覆盖模式写入需要确认
        if tool_name == "write_workspace_file":
            if args and args.get("mode") == "w":
                return True

        return False

    @staticmethod
    def get_risk_level(tool_name: str) -> RiskLevel:
        """获取工具的风险等级"""
        if tool_name in HIGH_RISK_TOOLS:
            return RiskLevel.HIGH
        elif tool_name in MEDIUM_RISK_TOOLS:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class SafetyChecker:
    """安全检查器"""

    @staticmethod
    def check_permission(agent_permission: AgentPermission, tool_name: str) -> bool:
        """检查 Agent 是否有权限调用该工具"""
        allowed_tools = PERMISSION_TOOLS.get(agent_permission, [])

        # Admin 拥有所有权限
        if "*" in allowed_tools:
            return True

        return tool_name in allowed_tools

    @staticmethod
    def is_high_risk(tool_name: str) -> bool:
        """判断是否为高风险操作"""
        return tool_name in HIGH_RISK_TOOLS

    @staticmethod
    def validate_command_safety(command: str) -> tuple[bool, str]:
        """
        验证命令安全性
        返回: (是否安全, 错误消息)
        """
        import re

        dangerous_patterns = [
            (r"\.\.", "相对路径越权"),
            (r"(?:^|\s|[<>|&;])/", "Unix 绝对路径"),
            (r"(?:^|\s|[<>|&;])~", "Unix 用户主目录"),
            (r"(?:^|\s|[<>|&;])\\", "Windows 根目录"),
            (r"(?i)(?:^|\s|[<>|&;])[a-z]:", "Windows 盘符"),
        ]

        for pattern, desc in dangerous_patterns:
            if re.search(pattern, command):
                return False, f"检测到危险操作: {desc}"

        return True, ""

    @staticmethod
    def get_agent_tools(agent_permission: AgentPermission) -> List[str]:
        """获取 Agent 允许的工具列表"""
        return PERMISSION_TOOLS.get(agent_permission, [])
