"""
SwarmMind 向量记忆存储

使用 ChromaDB 实现语义记忆存储和检索：
- 对话历史向量化存储
- 执行经验语义检索
- 用户偏好记忆
"""

import os
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """记忆条目"""
    id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    created_at: str = ""

    class Config:
        arbitrary_types_allowed = True


class VectorMemoryStore:
    """
    向量记忆存储

    使用 ChromaDB 进行语义存储和检索
    """

    def __init__(self, persist_dir: str, collection_name: str = "swarmmind_memory"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # 确保目录存在
        os.makedirs(persist_dir, exist_ok=True)

        # 延迟导入 chromadb
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "SwarmMind semantic memory"}
            )
            self._available = True
        except ImportError:
            self._available = False
            self.client = None
            self.collection = None

    @property
    def is_available(self) -> bool:
        """检查 ChromaDB 是否可用"""
        return self._available

    async def store(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存储记忆

        参数:
        - text: 要存储的文本内容
        - metadata: 元数据（如来源、类型、时间戳等）

        返回:
        - memory_id: 记忆 ID
        """
        if not self._available:
            return ""

        memory_id = str(uuid.uuid4())
        meta = metadata or {}
        meta["created_at"] = datetime.now().isoformat()

        try:
            self.collection.add(
                documents=[text],
                metadatas=[meta],
                ids=[memory_id]
            )
            return memory_id
        except Exception as e:
            print(f"[VectorMemory] Store error: {e}")
            return ""

    async def store_batch(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """批量存储记忆"""
        if not self._available or not texts:
            return []

        ids = [str(uuid.uuid4()) for _ in texts]

        if metadatas is None:
            metadatas = [{"created_at": datetime.now().isoformat()} for _ in texts]
        else:
            for meta in metadatas:
                meta["created_at"] = datetime.now().isoformat()

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            return ids
        except Exception as e:
            print(f"[VectorMemory] Batch store error: {e}")
            return []

    async def search(
        self,
        query: str,
        top_k: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[MemoryEntry]:
        """
        语义搜索记忆

        参数:
        - query: 查询文本
        - top_k: 返回结果数量
        - where_filter: 元数据过滤条件

        返回:
        - List[MemoryEntry]: 匹配的记忆条目列表
        """
        if not self._available:
            return []

        try:
            query_params = {
                "query_texts": [query],
                "n_results": top_k
            }

            if where_filter:
                query_params["where"] = where_filter

            results = self.collection.query(**query_params)

            entries = []
            if results and results.get("ids"):
                for i, doc_id in enumerate(results["ids"][0]):
                    text = results["documents"][0][i] if results.get("documents") else ""
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    distance = results["distances"][0][i] if results.get("distances") else 0

                    # 将距离转换为相似度分数（距离越小，分数越高）
                    score = 1.0 / (1.0 + distance)

                    entries.append(MemoryEntry(
                        id=doc_id,
                        text=text,
                        metadata=metadata,
                        score=score,
                        created_at=metadata.get("created_at", "")
                    ))

            return entries
        except Exception as e:
            print(f"[VectorMemory] Search error: {e}")
            return []

    async def get_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """根据 ID 获取记忆"""
        if not self._available:
            return None

        try:
            results = self.collection.get(ids=[memory_id])

            if results and results.get("ids"):
                return MemoryEntry(
                    id=results["ids"][0],
                    text=results["documents"][0] if results.get("documents") else "",
                    metadata=results["metadatas"][0] if results.get("metadatas") else {},
                    created_at=results["metadatas"][0].get("created_at", "") if results.get("metadatas") else ""
                )
            return None
        except Exception:
            return None

    async def delete(self, memory_id: str) -> bool:
        """删除记忆"""
        if not self._available:
            return False

        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    async def clear(self) -> bool:
        """清空所有记忆"""
        if not self._available:
            return False

        try:
            # 删除并重建集合
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "SwarmMind semantic memory"}
            )
            return True
        except Exception:
            return False

    async def count(self) -> int:
        """获取记忆总数"""
        if not self._available:
            return 0

        try:
            return self.collection.count()
        except Exception:
            return 0

    async def get_recent(
        self,
        limit: int = 10,
        memory_type: Optional[str] = None
    ) -> List[MemoryEntry]:
        """获取最近的记忆"""
        if not self._available:
            return []

        try:
            query_params = {"n_results": limit}

            if memory_type:
                query_params["where"] = {"type": memory_type}

            results = self.collection.peek(**query_params)

            entries = []
            if results and results.get("ids"):
                for i, doc_id in enumerate(results["ids"]):
                    text = results["documents"][i] if results.get("documents") else ""
                    metadata = results["metadatas"][i] if results.get("metadatas") else {}

                    entries.append(MemoryEntry(
                        id=doc_id,
                        text=text,
                        metadata=metadata,
                        score=1.0,
                        created_at=metadata.get("created_at", "")
                    ))

            return entries
        except Exception:
            return []


class ConversationMemoryStore(VectorMemoryStore):
    """对话记忆存储"""

    def __init__(self, persist_dir: str):
        super().__init__(persist_dir, "conversations")

    async def store_conversation(
        self,
        user_input: str,
        agent_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """存储对话"""
        text = f"User: {user_input}\nAgent: {agent_response}"
        meta = metadata or {}
        meta["type"] = "conversation"
        meta["user_input"] = user_input
        meta["agent_response"] = agent_response

        return await self.store(text, meta)

    async def find_similar_conversations(
        self,
        query: str,
        top_k: int = 5
    ) -> List[MemoryEntry]:
        """查找相似对话"""
        return await self.search(
            query,
            top_k=top_k,
            where_filter={"type": "conversation"}
        )


class ExperienceMemoryStore(VectorMemoryStore):
    """经验记忆存储"""

    def __init__(self, persist_dir: str):
        super().__init__(persist_dir, "experiences")

    async def store_experience(
        self,
        task: str,
        plan: str,
        result: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """存储执行经验"""
        text = f"Task: {task}\nPlan: {plan}\nResult: {result}"
        meta = metadata or {}
        meta["type"] = "experience"
        meta["task"] = task
        meta["success"] = success

        return await self.store(text, meta)

    async def find_similar_experiences(
        self,
        task: str,
        top_k: int = 5,
        success_only: bool = False
    ) -> List[MemoryEntry]:
        """查找相似任务经验"""
        where_filter = {"type": "experience"}
        if success_only:
            where_filter["success"] = True

        return await self.search(
            task,
            top_k=top_k,
            where_filter=where_filter
        )
