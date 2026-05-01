"""
SwarmMind 审计日志系统（增强版）

支持:
- 多 Agent 追踪
- 风险等级标记
- 用户确认记录
"""

import os
import json
import threading
import queue
import atexit
from datetime import datetime, timezone


class JSONLEventLogger:
    """单例模式审计日志器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, log_dir: str = "logs"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_logger(log_dir)
            return cls._instance

    def _init_logger(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_queue = queue.Queue()
        self._file_lock = threading.Lock()

        self.worker_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.worker_thread.start()

        atexit.register(self.shutdown)

    def _write_loop(self):
        while True:
            log_item = self.log_queue.get()

            if log_item is None:
                self.log_queue.task_done()
                break

            try:
                thread_id = log_item.get("thread_id", "system")
                safe_id = "".join(c for c in thread_id if c.isalnum() or c in "-_") or "default"
                file_path = os.path.join(self.log_dir, f"{safe_id}.jsonl")

                with self._file_lock:
                    with open(file_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_item, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Logger Error] 异步写日志失败: {e}")
            finally:
                self.log_queue.task_done()

    def log_event(
        self,
        thread_id: str,
        event: str,
        agent_name: str = "system",
        risk_level: str = "low",
        user_confirmed: bool = False,
        **kwargs
    ):
        """记录事件（增强版）"""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        log_item = {
            "ts": now_utc,
            "thread_id": thread_id,
            "event": event,
            "agent_name": agent_name,
            "risk_level": risk_level,
            "user_confirmed": user_confirmed,
            **kwargs
        }

        self.log_queue.put(log_item)

    def log_agent_action(self, agent_name: str, task, result, risk_level: str = "low"):
        """记录 Agent 动作"""
        self.log_event(
            thread_id="agent_actions",
            event="agent_action",
            agent_name=agent_name,
            risk_level=risk_level,
            task_summary=str(task)[:200],
            result_summary=str(result)[:200]
        )

    def shutdown(self):
        self.log_queue.put(None)
        self.log_queue.join()


# 全局审计日志器实例
audit_logger = JSONLEventLogger()
