"""
SwarmMind 上下文压缩器

使用 LLM 对长对话历史进行摘要压缩：
- 保留关键信息
- 减少上下文长度
- 支持 streaming 压缩
"""

from typing import List, Optional, Tuple
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage


# 压缩提示模板
COMPRESSION_PROMPT = """请将以下对话历史压缩为简洁的摘要，保留以下关键信息：
1. 用户的主要需求和目标
2. Agent 执行的关键操作和工具调用
3. 重要的决策和结论
4. 用户反馈和确认
5. 未完成的任务或待办事项

请用简洁的中文撰写摘要，控制在 200 字以内。

---
对话历史：
{conversation}
---

摘要："""


class ContextCompressor:
    """
    上下文压缩器

    当对话历史过长时，使用 LLM 进行智能压缩
    """

    def __init__(self, llm, compression_threshold: int = 10, keep_recent: int = 4):
        """
        初始化压缩器

        参数:
        - llm: LangChain LLM 实例
        - compression_threshold: 触发压缩的消息数量阈值
        - keep_recent: 保留的最近消息数量
        """
        self.llm = llm
        self.compression_threshold = compression_threshold
        self.keep_recent = keep_recent
        self._summary_cache: List[str] = []

    def needs_compression(self, messages: List[BaseMessage]) -> bool:
        """检查是否需要压缩"""
        non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        return len(non_system_msgs) > self.compression_threshold

    def split_messages(
        self,
        messages: List[BaseMessage]
    ) -> Tuple[Optional[SystemMessage], List[BaseMessage], List[BaseMessage]]:
        """
        分离消息为三部分：系统消息、待压缩消息、保留消息

        返回:
        - SystemMessage 或 None
        - 待压缩的旧消息列表
        - 保留的最近消息列表
        """
        # 提取系统消息
        system_msg = next(
            (m for m in messages if isinstance(m, SystemMessage)),
            None
        )

        # 非系统消息
        non_system_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        # 计算分割点
        if len(non_system_msgs) <= self.keep_recent:
            return system_msg, [], non_system_msgs

        split_point = len(non_system_msgs) - self.keep_recent
        old_messages = non_system_msgs[:split_point]
        recent_messages = non_system_msgs[split_point:]

        return system_msg, old_messages, recent_messages

    def messages_to_text(self, messages: List[BaseMessage]) -> str:
        """将消息列表转换为文本"""
        lines = []
        for msg in messages:
            role = "User" if isinstance(msg, HumanMessage) else "Agent"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            # 截断过长的消息
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    async def compress(
        self,
        messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        压缩消息列表

        参数:
        - messages: 原始消息列表

        返回:
        - 压缩后的消息列表
        """
        if not self.needs_compression(messages):
            return messages

        system_msg, old_messages, recent_messages = self.split_messages(messages)

        if not old_messages:
            return messages

        # 构建压缩请求
        conversation_text = self.messages_to_text(old_messages)
        prompt = COMPRESSION_PROMPT.format(conversation=conversation_text)

        try:
            # 调用 LLM 生成摘要
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            summary = response.content if hasattr(response, "content") else str(response)

            # 缓存摘要
            self._summary_cache.append(summary)

            # 构建压缩后的消息列表
            compressed_messages = []

            # 添加系统消息
            if system_msg:
                compressed_messages.append(system_msg)

            # 添加历史摘要作为系统消息
            summary_text = f"[历史对话摘要]\n{summary}"
            compressed_messages.append(SystemMessage(content=summary_text))

            # 添加保留的最近消息
            compressed_messages.extend(recent_messages)

            return compressed_messages

        except Exception as e:
            # 压缩失败，返回原始消息
            print(f"[ContextCompressor] Compression failed: {e}")
            return messages

    def compress_sync(
        self,
        messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """同步压缩方法"""
        if not self.needs_compression(messages):
            return messages

        system_msg, old_messages, recent_messages = self.split_messages(messages)

        if not old_messages:
            return messages

        conversation_text = self.messages_to_text(old_messages)
        prompt = COMPRESSION_PROMPT.format(conversation=conversation_text)

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            summary = response.content if hasattr(response, "content") else str(response)

            self._summary_cache.append(summary)

            compressed_messages = []
            if system_msg:
                compressed_messages.append(system_msg)

            summary_text = f"[历史对话摘要]\n{summary}"
            compressed_messages.append(SystemMessage(content=summary_text))
            compressed_messages.extend(recent_messages)

            return compressed_messages

        except Exception as e:
            print(f"[ContextCompressor] Compression failed: {e}")
            return messages

    def get_summary_history(self) -> List[str]:
        """获取所有生成的摘要历史"""
        return self._summary_cache.copy()

    def clear_cache(self) -> None:
        """清空摘要缓存"""
        self._summary_cache.clear()


def trim_and_compress_messages(
    messages: List[BaseMessage],
    trigger_turns: int = 10,
    keep_turns: int = 4,
    compressor: Optional[ContextCompressor] = None
) -> Tuple[List[BaseMessage], bool]:
    """
    智能修剪和压缩消息

    参数:
    - messages: 原始消息列表
    - trigger_turns: 触发压缩的轮次阈值
    - keep_turns: 保留的最近轮次
    - compressor: 可选的压缩器实例

    返回:
    - 压缩后的消息列表
    - 是否进行了压缩
    """
    # 计算用户轮次
    user_messages = [m for m in messages if isinstance(m, HumanMessage)]

    if len(user_messages) <= trigger_turns:
        return messages, False

    # 如果提供了压缩器，使用 LLM 压缩
    if compressor:
        compressed = compressor.compress_sync(messages)
        return compressed, True

    # 否则使用简单的轮次裁剪
    from .memory import trim_context_messages
    trimmed, _ = trim_context_messages(messages, trigger_turns, keep_turns)
    return trimmed, True
