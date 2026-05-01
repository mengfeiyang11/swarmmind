"""
SwarmMind 核心模块
"""

from .memory import MemorySystem, AgentState, trim_context_messages
from .provider import get_provider
from .logger import audit_logger
from .security import AgentPermission, SafetyChecker, ConfirmProtocol, RiskLevel
from .policy import PolicyEngine, PolicyRule, PolicyAction, get_default_policy_engine
from .vector_memory import VectorMemoryStore, MemoryEntry
from .compressor import ContextCompressor, trim_and_compress_messages
from .experience import ExperienceStore, Experience
from .anomaly import BehaviorMonitor, AnomalyReport, check_before_action

# 延迟导入 orchestrator 以避免循环导入
def get_orchestrator():
    from .orchestrator import SafeOrchestrator
    return SafeOrchestrator

__all__ = [
    "MemorySystem",
    "AgentState",
    "trim_context_messages",
    "get_provider",
    "audit_logger",
    "AgentPermission",
    "SafetyChecker",
    "ConfirmProtocol",
    "RiskLevel",
    "get_orchestrator",
    "PolicyEngine",
    "PolicyRule",
    "PolicyAction",
    "get_default_policy_engine",
    "VectorMemoryStore",
    "MemoryEntry",
    "ContextCompressor",
    "trim_and_compress_messages",
    "ExperienceStore",
    "Experience",
    "BehaviorMonitor",
    "AnomalyReport",
    "check_before_action",
]
