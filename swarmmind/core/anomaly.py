"""
SwarmMind 异常行为检测系统

监控 Agent 行为，检测异常偏离：
- 短时间大量相同工具调用
- 尝试调用未授权工具
- 敏感文件读取模式
- 参数注入攻击检测
"""

import re
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from pydantic import BaseModel, Field
from enum import Enum


class AnomalySeverity(str, Enum):
    """异常严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendedAction(str, Enum):
    """推荐动作"""
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class AnomalyReport(BaseModel):
    """异常报告"""
    is_anomaly: bool = False
    severity: AnomalySeverity = AnomalySeverity.LOW
    description: str = ""
    recommended_action: RecommendedAction = RecommendedAction.ALLOW
    details: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = ""

    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class AgentAction(BaseModel):
    """Agent 行动记录"""
    agent: str
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    timestamp: float = 0.0
    duration: float = 0.0

    def __init__(self, **data):
        super().__init__(**data)
        if not self.timestamp:
            self.timestamp = time.time()


class BehaviorMonitor:
    """
    行为监控器

    实时监控 Agent 行为，检测异常模式
    """

    def __init__(
        self,
        history_window: int = 300,      # 历史窗口（秒）
        max_same_tool_calls: int = 10,   # 同一工具最大调用次数
        max_total_calls: int = 50,       # 总调用次数上限
        loop_detection_window: int = 30  # 循环检测窗口（秒）
    ):
        self.history_window = history_window
        self.max_same_tool_calls = max_same_tool_calls
        self.max_total_calls = max_total_calls
        self.loop_detection_window = loop_detection_window

        # 行动历史
        self._action_history: List[AgentAction] = []

        # 统计数据
        self._tool_call_counts: Dict[str, int] = defaultdict(int)
        self._agent_call_counts: Dict[str, int] = defaultdict(int)

        # 异常记录
        self._anomaly_history: List[AnomalyReport] = []

    def record_action(
        self,
        agent: str,
        tool: str,
        args: Dict[str, Any],
        result: str = ""
    ) -> None:
        """记录 Agent 行动"""
        action = AgentAction(
            agent=agent,
            tool=tool,
            args=args,
            result_summary=result[:500] if result else "",
            timestamp=time.time()
        )

        self._action_history.append(action)

        # 更新计数
        self._tool_call_counts[tool] += 1
        self._agent_call_counts[agent] += 1

        # 清理旧记录
        self._cleanup_old_actions()

    def _cleanup_old_actions(self) -> None:
        """清理过期的行动记录（同步清理计数器）"""
        cutoff_time = time.time() - self.history_window

        # 找出需要移除的行动
        to_remove = []
        for action in self._action_history:
            if action.timestamp <= cutoff_time:
                to_remove.append(action)

        # 同步递减计数器
        for action in to_remove:
            if self._tool_call_counts[action.tool] > 0:
                self._tool_call_counts[action.tool] -= 1
            if self._agent_call_counts[action.agent] > 0:
                self._agent_call_counts[action.agent] -= 1

        # 移除过期记录
        self._action_history = [
            action for action in self._action_history
            if action.timestamp > cutoff_time
        ]

    def check_anomaly(
        self,
        agent: str,
        tool: str,
        args: Dict[str, Any]
    ) -> AnomalyReport:
        """
        检测异常行为

        返回:
        - AnomalyReport: 异常报告
        """
        # 1. 检测工具调用循环
        loop_check = self._check_tool_loop(tool)
        if loop_check.is_anomaly:
            self._record_anomaly(loop_check)
            return loop_check

        # 2. 检测高频调用
        frequency_check = self._check_call_frequency(tool)
        if frequency_check.is_anomaly:
            self._record_anomaly(frequency_check)
            return frequency_check

        # 3. 检测危险参数
        args_check = self._check_dangerous_args(args)
        if args_check.is_anomaly:
            self._record_anomaly(args_check)
            return args_check

        # 4. 检测敏感文件访问模式
        if "file" in tool or "read" in tool or "write" in tool:
            file_check = self._check_sensitive_file_access(args)
            if file_check.is_anomaly:
                self._record_anomaly(file_check)
                return file_check

        # 5. 检测注入攻击模式
        injection_check = self._check_injection_patterns(args)
        if injection_check.is_anomaly:
            self._record_anomaly(injection_check)
            return injection_check

        # 正常行为
        return AnomalyReport(
            is_anomaly=False,
            severity=AnomalySeverity.LOW,
            description="行为正常",
            recommended_action=RecommendedAction.ALLOW
        )

    def _check_tool_loop(self, tool: str) -> AnomalyReport:
        """检测工具调用循环"""
        window_start = time.time() - self.loop_detection_window
        recent_calls = [
            a for a in self._action_history
            if a.timestamp > window_start and a.tool == tool
        ]

        if len(recent_calls) >= self.max_same_tool_calls:
            return AnomalyReport(
                is_anomaly=True,
                severity=AnomalySeverity.HIGH,
                description=f"检测到工具循环调用: {tool} 在 {self.loop_detection_window}秒内被调用 {len(recent_calls)} 次",
                recommended_action=RecommendedAction.BLOCK,
                details={
                    "tool": tool,
                    "call_count": len(recent_calls),
                    "window_seconds": self.loop_detection_window
                }
            )

        return AnomalyReport(is_anomaly=False)

    def _check_call_frequency(self, tool: str) -> AnomalyReport:
        """检测调用频率异常"""
        total_calls = len(self._action_history)

        if total_calls >= self.max_total_calls:
            return AnomalyReport(
                is_anomaly=True,
                severity=AnomalySeverity.MEDIUM,
                description=f"调用频率异常: 在 {self.history_window}秒内总调用次数达到 {total_calls}",
                recommended_action=RecommendedAction.WARN,
                details={
                    "total_calls": total_calls,
                    "window_seconds": self.history_window
                }
            )

        return AnomalyReport(is_anomaly=False)

    def _check_dangerous_args(self, args: Dict[str, Any]) -> AnomalyReport:
        """检测危险参数"""
        # 检测命令注入
        dangerous_patterns = [
            (r";\s*rm\s", "可能删除文件"),
            (r";\s*cat\s+/etc/", "尝试读取系统文件"),
            (r"\|\s*sh\b", "管道到 shell"),
            (r"`.*`", "命令替换"),
            (r"\$\([^)]+\)", "命令替换"),
            (r"&&\s*rm", "条件删除"),
            (r">\s*/etc/", "重定向到系统文件"),
        ]

        args_str = str(args)

        for pattern, desc in dangerous_patterns:
            if re.search(pattern, args_str, re.IGNORECASE):
                return AnomalyReport(
                    is_anomaly=True,
                    severity=AnomalySeverity.CRITICAL,
                    description=f"检测到危险参数: {desc}",
                    recommended_action=RecommendedAction.BLOCK,
                    details={
                        "pattern": pattern,
                        "description": desc,
                        "args_preview": args_str[:200]
                    }
                )

        return AnomalyReport(is_anomaly=False)

    def _check_sensitive_file_access(self, args: Dict[str, Any]) -> AnomalyReport:
        """检测敏感文件访问"""
        sensitive_patterns = [
            (r"\.env", "环境配置文件"),
            (r"\.pem", "证书文件"),
            (r"\.key", "密钥文件"),
            (r"id_rsa", "SSH 私钥"),
            (r"credentials", "凭据文件"),
            (r"password", "密码文件"),
            (r"secret", "机密文件"),
            (r"/etc/shadow", "系统密码文件"),
            (r"/etc/passwd", "用户信息文件"),
        ]

        for key, value in args.items():
            if isinstance(value, str):
                for pattern, desc in sensitive_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        return AnomalyReport(
                            is_anomaly=True,
                            severity=AnomalySeverity.HIGH,
                            description=f"检测到敏感文件访问: {desc}",
                            recommended_action=RecommendedAction.BLOCK,
                            details={
                                "pattern": pattern,
                                "description": desc,
                                "arg_key": key,
                                "arg_value": value
                            }
                        )

        return AnomalyReport(is_anomaly=False)

    def _check_injection_patterns(self, args: Dict[str, Any]) -> AnomalyReport:
        """检测注入攻击模式"""
        injection_patterns = [
            (r"base64", "Base64 编码（可能混淆恶意代码）"),
            (r"\\x[0-9a-f]{2}", "十六进制编码（可能混淆）"),
            (r"%[0-9a-f]{2}", "URL 编码（可能绕过检测）"),
            (r"eval\s*\(", "动态代码执行"),
            (r"exec\s*\(", "动态代码执行"),
            (r"__import__", "动态导入"),
        ]

        args_str = str(args)

        for pattern, desc in injection_patterns:
            if re.search(pattern, args_str, re.IGNORECASE):
                # 计算严重程度
                severity = AnomalySeverity.HIGH if "eval" in args_str or "exec" in args_str else AnomalySeverity.MEDIUM

                return AnomalyReport(
                    is_anomaly=True,
                    severity=severity,
                    description=f"检测到可疑模式: {desc}",
                    recommended_action=RecommendedAction.WARN,
                    details={
                        "pattern": pattern,
                        "description": desc
                    }
                )

        return AnomalyReport(is_anomaly=False)

    def _record_anomaly(self, report: AnomalyReport) -> None:
        """记录异常"""
        self._anomaly_history.append(report)

        # 只保留最近 100 条异常记录
        if len(self._anomaly_history) > 100:
            self._anomaly_history = self._anomaly_history[-100:]

    def get_statistics(self) -> Dict[str, Any]:
        """获取监控统计"""
        return {
            "total_actions": len(self._action_history),
            "tool_call_counts": dict(self._tool_call_counts),
            "agent_call_counts": dict(self._agent_call_counts),
            "anomaly_count": len(self._anomaly_history),
            "window_seconds": self.history_window
        }

    def get_recent_anomalies(self, limit: int = 10) -> List[AnomalyReport]:
        """获取最近的异常记录"""
        return self._anomaly_history[-limit:]

    def clear_history(self) -> None:
        """清空历史记录"""
        self._action_history.clear()
        self._tool_call_counts.clear()
        self._agent_call_counts.clear()


# 全局监控器实例
behavior_monitor = BehaviorMonitor()


def check_before_action(
    agent: str,
    tool: str,
    args: Dict[str, Any]
) -> AnomalyReport:
    """
    行动前检查（便捷函数）

    参数:
    - agent: Agent 名称
    - tool: 工具名称
    - args: 工具参数

    返回:
    - AnomalyReport: 异常报告
    """
    return behavior_monitor.check_anomaly(agent, tool, args)


def record_action(
    agent: str,
    tool: str,
    args: Dict[str, Any],
    result: str = ""
) -> None:
    """记录行动（便捷函数）"""
    behavior_monitor.record_action(agent, tool, args, result)
