"""
SwarmMind 配置模块
"""

import os
from dotenv import load_dotenv

load_dotenv()

CORE_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(CORE_DIR)
PROJECT_ROOT = os.path.dirname(PACKAGE_DIR)

WORKSPACE_DIR = os.getenv("SWARMIND_WORKSPACE", os.path.join(PROJECT_ROOT, "workspace"))

# 数据库路径
DB_PATH = os.path.join(WORKSPACE_DIR, "state.sqlite3")

# 记忆系统
MEMORY_DIR = os.path.join(WORKSPACE_DIR, "memory")

# 人设区
PERSONAS_DIR = os.path.join(WORKSPACE_DIR, "personas")

# 脚本区
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")

# 工作区（唯一被允许执行文件与 shell 操作的空间）
WORKSPACE_OFFICE_DIR = os.path.join(WORKSPACE_DIR, "office")

# 技能目录
SKILLS_DIR = os.path.join(WORKSPACE_OFFICE_DIR, "skills")

# 日志目录
LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")

# 任务文件
TASKS_FILE = os.path.join(WORKSPACE_DIR, "tasks.json")

# 确保目录存在
for d in [WORKSPACE_DIR, MEMORY_DIR, PERSONAS_DIR, SCRIPTS_DIR, WORKSPACE_OFFICE_DIR, SKILLS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)
