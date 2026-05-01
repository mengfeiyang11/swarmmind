"""
SwarmMind 安全策略引擎

提供细粒度的基于属性的访问控制（ABAC）：
- 基于角色的权限控制
- 基于资源的权限控制
- 基于操作类型的权限控制
- 可配置的策略规则
"""

import re
import os
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

import yaml


class PolicyAction(str, Enum):
    """策略动作"""
    ALLOW = "allow"
    DENY = "deny"
    CONFIRM = "confirm"


class PolicyRule(BaseModel):
    """策略规则"""
    name: str
    agent_role: str                    # "planner" | "executor" | "reviewer" | "*"
    tool_pattern: str                  # 支持通配符 "write_*", "execute_*", "*"
    resource_pattern: str               # "workspace/office/**" 或 "!workspace/office/**"（否定）
    action: PolicyAction
    condition: str = ""                 # 可选条件表达式 "file_size < 1MB"
    priority: int = 50                  # 优先级，数字越大越优先
    description: str = ""               # 规则描述

    class Config:
        use_enum_values = True


class PolicyDecision(BaseModel):
    """策略决策结果"""
    action: PolicyAction
    matched_rule: Optional[str] = None
    reason: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)


class PolicyEngine:
    """
    安全策略引擎

    支持：
    - 多规则优先级排序
    - 通配符模式匹配
    - 条件表达式求值
    - YAML 配置文件加载
    """

    def __init__(self, rules: Optional[List[PolicyRule]] = None):
        self.rules: List[PolicyRule] = []
        if rules:
            self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)

    def add_rule(self, rule: PolicyRule) -> None:
        """添加规则并重新排序"""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)

    def evaluate(
        self,
        agent_role: str,
        tool: str,
        resource: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> PolicyDecision:
        """
        评估策略决策

        参数：
        - agent_role: Agent 角色
        - tool: 工具名称
        - resource: 资源路径（可选）
        - context: 额外上下文（如文件大小、操作参数等）

        返回：
        - PolicyDecision: 包含决策动作和匹配的规则信息
        """
        if context is None:
            context = {}

        for rule in self.rules:
            if self._match_rule(rule, agent_role, tool, resource, context):
                return PolicyDecision(
                    action=rule.action,
                    matched_rule=rule.name,
                    reason=f"Matched rule: {rule.name} (priority={rule.priority})",
                    context={
                        "agent_role": agent_role,
                        "tool": tool,
                        "resource": resource
                    }
                )

        return PolicyDecision(
            action=PolicyAction.DENY,
            matched_rule=None,
            reason="No matching rule, default deny",
            context={"agent_role": agent_role, "tool": tool}
        )

    def _match_rule(
        self,
        rule: PolicyRule,
        agent_role: str,
        tool: str,
        resource: str,
        context: Dict[str, Any]
    ) -> bool:
        """检查规则是否匹配"""
        # 1. 检查 Agent 角色
        if rule.agent_role != "*" and rule.agent_role != agent_role:
            return False

        # 2. 检查工具模式
        if not self._match_pattern(rule.tool_pattern, tool):
            return False

        # 3. 检查资源模式
        if rule.resource_pattern and resource:
            if not self._match_resource_pattern(rule.resource_pattern, resource):
                return False

        # 4. 检查条件表达式
        if rule.condition and not self._evaluate_condition(rule.condition, context):
            return False

        return True

    def _match_pattern(self, pattern: str, value: str) -> bool:
        """通配符模式匹配（安全版：转义正则元字符）"""
        if pattern == "*":
            return True

        # 安全转换：先转义正则元字符，再替换通配符
        # 转义顺序很重要：先转义所有特殊字符，再还原通配符语义
        regex_pattern = ""
        for char in pattern:
            if char == "*":
                regex_pattern += ".*"
            elif char == "?":
                regex_pattern += "."
            elif char in ".+()[]{}^$|\\":
                regex_pattern += "\\" + char
            else:
                regex_pattern += char

        regex_pattern = f"^{regex_pattern}$"

        return bool(re.match(regex_pattern, value, re.IGNORECASE))

    def _match_resource_pattern(self, pattern: str, resource: str) -> bool:
        """资源路径模式匹配（安全版）"""
        # 支持否定模式：!workspace/office/** 表示"不在此路径下"
        if pattern.startswith("!"):
            actual_pattern = pattern[1:]
            # 否定匹配：如果资源匹配模式，则拒绝
            return not self._match_resource_pattern(actual_pattern, resource)

        # 标准 glob 模式匹配
        if pattern.endswith("/**"):
            base_pattern = pattern[:-3]
            return resource.startswith(base_pattern)

        # 精确匹配
        if pattern == resource:
            return True

        # glob 模式转换为正则（安全版：正确转义）
        regex_pattern = pattern.replace("**", ".*")

        # 逐字符处理以安全转义正则元字符
        result = []
        i = 0
        while i < len(regex_pattern):
            ch = regex_pattern[i]
            if ch == "*":
                result.append("[^/]*")
            elif ch == "?":
                result.append(".")
            elif ch in ".+()[]{}^$|\\":
                result.append("\\" + ch)
            else:
                result.append(ch)
            i += 1

        regex_pattern = "".join(result)
        regex_pattern = f"^{regex_pattern}"

        return bool(re.match(regex_pattern, resource, re.IGNORECASE))

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """求值条件表达式"""
        if not condition:
            return True

        try:
            # 简单表达式求值（安全限制）
            # 支持的比较操作符：<, >, <=, >=, ==, !=
            safe_ops = ["<", ">", "<=", ">=", "==", "!="]

            for op in safe_ops:
                if op in condition:
                    parts = condition.split(op)
                    if len(parts) == 2:
                        left = parts[0].strip()
                        right = parts[1].strip()

                        # 从上下文获取值
                        left_val = context.get(left, left)
                        right_val = context.get(right, right)

                        # 尝试数值比较
                        try:
                            left_num = float(left_val)
                            right_num = float(right_val)

                            if op == "<":
                                return left_num < right_num
                            elif op == ">":
                                return left_num > right_num
                            elif op == "<=":
                                return left_num <= right_num
                            elif op == ">=":
                                return left_num >= right_num
                            elif op == "==":
                                return left_num == right_num
                            elif op == "!=":
                                return left_num != right_num
                        except (ValueError, TypeError):
                            # 字符串比较
                            return str(left_val) == str(right_val) if op in ["==", "!="] else False

            return True
        except Exception:
            return True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PolicyEngine":
        """从 YAML 文件加载策略"""
        engine = cls()

        if not os.path.exists(yaml_path):
            return engine

        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if config and "rules" in config:
            for rule_data in config["rules"]:
                rule = PolicyRule(
                    name=rule_data.get("name", "unnamed"),
                    agent_role=rule_data.get("agent_role", "*"),
                    tool_pattern=rule_data.get("tool_pattern", "*"),
                    resource_pattern=rule_data.get("resource_pattern", ""),
                    action=PolicyAction(rule_data.get("action", "allow")),
                    condition=rule_data.get("condition", ""),
                    priority=rule_data.get("priority", 50),
                    description=rule_data.get("description", "")
                )
                engine.add_rule(rule)

        return engine

    def to_yaml(self, yaml_path: str) -> None:
        """导出策略到 YAML 文件"""
        rules_data = []
        for rule in self.rules:
            rules_data.append({
                "name": rule.name,
                "agent_role": rule.agent_role,
                "tool_pattern": rule.tool_pattern,
                "resource_pattern": rule.resource_pattern,
                "action": rule.action.value if isinstance(rule.action, PolicyAction) else rule.action,
                "condition": rule.condition,
                "priority": rule.priority,
                "description": rule.description
            })

        config = {"rules": rules_data}

        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def get_default_policy_engine() -> PolicyEngine:
    """获取默认策略引擎实例"""
    from .config import MEMORY_DIR

    default_policy_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "policies",
        "default.yaml"
    )

    if os.path.exists(default_policy_path):
        return PolicyEngine.from_yaml(default_policy_path)

    # 如果没有配置文件，返回内置默认策略
    engine = PolicyEngine()

    # 默认规则
    default_rules = [
        PolicyRule(
            name="deny_outside_workspace",
            agent_role="*",
            tool_pattern="*",
            resource_pattern="!workspace/office/**",
            action=PolicyAction.DENY,
            priority=100,
            description="禁止访问工作区外的资源"
        ),
        PolicyRule(
            name="confirm_shell_execute",
            agent_role="executor",
            tool_pattern="execute_workspace_shell",
            resource_pattern="*",
            action=PolicyAction.CONFIRM,
            priority=90,
            description="Shell 命令执行需要确认"
        ),
        PolicyRule(
            name="confirm_delete",
            agent_role="executor",
            tool_pattern="delete_*",
            resource_pattern="*",
            action=PolicyAction.DENY,
            priority=85,
            description="删除操作默认拒绝（安全策略）"
        ),
        PolicyRule(
            name="confirm_write_file",
            agent_role="executor",
            tool_pattern="write_workspace_file",
            resource_pattern="*",
            action=PolicyAction.CONFIRM,
            priority=50,
            description="文件写入需要确认"
        ),
    ]

    for rule in default_rules:
        engine.add_rule(rule)

    return engine
