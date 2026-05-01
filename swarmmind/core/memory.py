"""
SwarmMind 双水位记忆系统

长期记忆: user_profile.md (用户画像)
短期记忆: SQLite 数据库 (对话历史)
"""

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
import os


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

    def __init__(self, workspace_dir: str):
        # workspace_dir 已经是 memory 目录的父目录
        # 所以 profile 路径应该是 workspace_dir/user_profile.md 而不是 workspace_dir/memory/user_profile.md
        self.profile_path = os.path.join(workspace_dir, "user_profile.md")
        self.db_path = os.path.join(workspace_dir, "state.sqlite3")
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)

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
        with open(self.profile_path, "w", encoding="utf-8") as f:
            f.write(content)
        return "记忆档案已成功更新。"
