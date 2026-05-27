"""
SwarmMind 双水位记忆系统

长期记忆: user_profile.md (用户画像)
短期记忆: SQLite 数据库 (对话历史)
"""

from typing import Annotated, TypedDict, Optional
from datetime import datetime, timezone
import re
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
import os

from .logger import audit_logger
from .provider import get_provider
from .config import AUTO_PROFILE_UPDATE, PROFILE_MAX_CHARS

PROFILE_SECTIONS = [
    "基本信息",
    "偏好",
    "约束与禁忌",
    "最近上下文",
    "变更记录"
]

PROFILE_TEMPLATE = """# 用户画像
最后更新: {timestamp}

## 基本信息
- 暂无

## 偏好
- 暂无

## 约束与禁忌
- 暂无

## 最近上下文
- 暂无

## 变更记录
- {timestamp} 初始化
"""

PROFILE_UPDATE_PROMPT = """你是用户画像维护助手，需要基于本轮对话更新用户画像。
请严格遵守以下规则：
1. 仅从本轮对话中抽取“长期有效、可复用、无隐私风险”的信息。
2. 不写入一次性任务、临时指令、敏感数据（API Key、密码、Token 等）。
3. 若信息冲突，以本轮为准，必要时保留旧信息的简要说明。
4. 保持画像结构固定，且顺序必须如下：
   - 基本信息
   - 偏好
   - 约束与禁忌
   - 最近上下文
   - 变更记录
5. 请更新“最后更新”时间为 {timestamp}。
6. 若本轮无可更新内容，请在“变更记录”中记录“无新增”，其余内容保持不变。
7. 输出必须是完整的 markdown 内容，不要额外解释或代码块。
8. 总长度不超过 {max_chars} 字符。

【现有用户画像】
{current_profile}

【本轮对话】
用户: {user_input}
助手: {assistant_output}

【当前上下文】
工作区: {workspace_dir}
办公区: {office_dir}
项目根: {project_root}
时间戳: {timestamp}
"""

SENSITIVE_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN[ A-Z]*PRIVATE KEY-----",
    r"(?i)api[_-]?key\\s*[:=]\\s*\\S+",
    r"(?i)password\\s*[:=]\\s*\\S+",
    r"(?i)secret\\s*[:=]\\s*\\S+",
    r"(?i)token\\s*[:=]\\s*\\S+"
]

class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str


def trim_context_messages(
    messages: list[BaseMessage],
    trigger_turns: int = 8,
    keep_turns: int = 4
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """
    按照完整用户回合来裁剪上下文。
    一个回合从 HumanMessage 开始，直到下一个 HumanMessage 结束。
    会把 AIMessage、tool_calls、ToolMessage 一并保留。
    """
    first_system = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

    if not non_system_msgs:
        return ([first_system] if first_system else []), []

    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []

    for msg in non_system_msgs:
        if isinstance(msg, HumanMessage):
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            if current_turn:
                current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    total_turns = len(turns)

    if total_turns < trigger_turns:
        final_messages = ([first_system] if first_system else []) + non_system_msgs
        return final_messages, []

    recent_turns = turns[-keep_turns:]
    discarded_turns = turns[:-keep_turns]

    final_messages: list[BaseMessage] = []
    if first_system:
        final_messages.append(first_system)
    for turn in recent_turns:
        final_messages.extend(turn)

    discarded_messages: list[BaseMessage] = []
    for turn in discarded_turns:
        discarded_messages.extend(turn)

    return final_messages, discarded_messages


class MemorySystem:
    """双水位记忆系统"""

    def __init__(
        self,
        workspace_dir: str,
        llm: Optional[object] = None,
        auto_update: Optional[bool] = None,
        max_chars: Optional[int] = None
    ):
        # workspace_dir 为记忆存储目录
        self.profile_path = os.path.join(workspace_dir, "user_profile.md")
        self.db_path = os.path.join(workspace_dir, "state.sqlite3")
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        self.auto_update = AUTO_PROFILE_UPDATE if auto_update is None else auto_update
        self.max_chars = PROFILE_MAX_CHARS if max_chars is None else max_chars
        self._llm = llm

    def set_llm(self, llm: object) -> None:
        """设置用于画像更新的 LLM"""
        self._llm = llm

    def load_user_profile(self) -> str:
        """加载用户画像"""
        if os.path.exists(self.profile_path):
            with open(self.profile_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return content
        return "暂无记录"

    def save_user_profile(self, content: str) -> str:
        """保存用户画像"""
        tmp_path = f"{self.profile_path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp_path, self.profile_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return "记忆档案已成功更新。"

    def update_user_profile(
        self,
        user_input: str,
        assistant_output: str,
        context: Optional[dict] = None
    ) -> bool:
        """基于本轮对话更新用户画像"""
        if not self.auto_update:
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_skipped",
                agent_name="memory",
                risk_level="low",
                reason="auto_update_disabled"
            )
            return False

        if not user_input or not assistant_output:
            return False

        llm = self._llm or self._build_llm()
        if not llm:
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_skipped",
                agent_name="memory",
                risk_level="low",
                reason="llm_unavailable"
            )
            return False

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        current_profile = self.load_user_profile()
        if current_profile == "暂无记录":
            current_profile = PROFILE_TEMPLATE.format(timestamp=timestamp)

        context = context or {}
        prompt = PROFILE_UPDATE_PROMPT.format(
            timestamp=timestamp,
            max_chars=self.max_chars,
            current_profile=current_profile,
            user_input=self._truncate_text(user_input),
            assistant_output=self._truncate_text(assistant_output),
            workspace_dir=context.get("workspace_dir", ""),
            office_dir=context.get("office_dir", ""),
            project_root=context.get("project_root", "")
        )

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            updated_profile = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_failed",
                agent_name="memory",
                risk_level="low",
                error=str(e)[:200]
            )
            return False

        updated_profile = self._strip_code_fences(updated_profile.strip())

        if not updated_profile:
            return False

        if len(updated_profile) > self.max_chars:
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_rejected",
                agent_name="memory",
                risk_level="low",
                reason="profile_too_long",
                size=len(updated_profile)
            )
            return False

        if self._contains_sensitive(updated_profile):
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_rejected",
                agent_name="memory",
                risk_level="medium",
                reason="sensitive_content_detected"
            )
            return False

        if not self._has_required_sections(updated_profile):
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_rejected",
                agent_name="memory",
                risk_level="low",
                reason="missing_sections"
            )
            return False

        try:
            self.save_user_profile(updated_profile)
        except Exception as e:
            try:
                self.save_user_profile(current_profile)
            except Exception:
                pass
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_failed",
                agent_name="memory",
                risk_level="low",
                error=str(e)[:200]
            )
            return False

        audit_logger.log_event(
            thread_id="memory",
            event="profile_update_success",
            agent_name="memory",
            risk_level="low"
        )
        return True

    def _build_llm(self) -> Optional[object]:
        provider = os.getenv("DEFAULT_PROVIDER", "openai")
        model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
        try:
            return get_provider(provider_name=provider, model_name=model)
        except Exception as e:
            audit_logger.log_event(
                thread_id="memory",
                event="profile_update_failed",
                agent_name="memory",
                risk_level="low",
                error=f"llm_init_failed:{str(e)[:150]}"
            )
            return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if text.startswith("```"):
            lines = text.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return text

    @staticmethod
    def _truncate_text(text: str, max_len: int = 1200) -> str:
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text

    @staticmethod
    def _contains_sensitive(text: str) -> bool:
        return any(re.search(pattern, text) for pattern in SENSITIVE_PATTERNS)

    @staticmethod
    def _has_required_sections(text: str) -> bool:
        if "最后更新:" not in text:
            return False
        return all(f"## {section}" in text for section in PROFILE_SECTIONS)
