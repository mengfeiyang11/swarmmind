# SwarmMind - 多智能体协作框架

<div align="center">

**安全、可控、智能的多 Agent 协作系统**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://github.com/langchain-ai/langchain)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 项目简介

SwarmMind 是一个基于 LangChain/LangGraph 构建的多智能体协作框架，专为技术研究和自动化任务设计。系统采用 **Planner → Executor → Reviewer** 三阶段协作模式，具备完善的沙箱隔离、权限控制和审计日志能力。

### 核心特性

| 特性 | 描述 |
|------|------|
| **多 Agent 协作** | Planner（规划）、Executor（执行）、Reviewer（审查）三角色协作 |
| **沙箱安全隔离** | 文件操作限制在 workspace 目录，符号链接解析 + 路径白名单防护 |
| **向量语义记忆** | ChromaDB 向量存储，支持语义检索历史经验 |
| **上下文智能压缩** | LLM 驱动的对话摘要，优化长对话处理 |
| **代码执行沙箱** | AST 级别的代码安全分析，安全执行 Python/JavaScript 代码 |
| **API 调用工具** | 标准化 REST API 调用 |
| **异常行为检测** | 实时监控 Agent 行为，检测注入攻击和异常模式 |
| **经验回放系统** | 记录执行经验，相似任务智能检索 |
| **并行任务执行** | 支持独立任务并行执行，提升效率 |
| **多 LLM 支持** | OpenAI、Anthropic、阿里云、腾讯、智谱、Ollama |

---

## 目录结构

```
Swarmmind/
├── swarmmind/                    # 主包
│   ├── __init__.py              # 包入口
│   ├── core/                    # 核心模块
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # 多 Agent 编排器
│   │   ├── security.py          # 权限和安全检查
│   │   ├── memory.py            # 双层记忆系统
│   │   ├── vector_memory.py     # 向量语义记忆
│   │   ├── compressor.py        # 上下文压缩器
│   │   ├── experience.py        # 经验回放系统
│   │   ├── anomaly.py           # 异常行为检测
│   │   ├── provider.py          # LLM 提供商适配
│   │   ├── logger.py            # 审计日志
│   │   ├── config.py            # 配置管理
│   │   └── tools/               # 工具模块
│   │       ├── __init__.py
│   │       ├── base.py          # 工具基类
│   │       ├── sandbox.py       # 沙箱文件操作
│   │       ├── code_sandbox.py  # 代码执行沙箱
│   │       ├── api_tool.py      # API 调用工具
│   │       └── builtins.py      # 内置工具集
│   ├── agents/                  # Agent 模块
│   │   ├── __init__.py
│   │   ├── base.py              # Agent 基类
│   │   ├── planner.py           # 规划 Agent
│   │   ├── executor.py          # 执行 Agent
│   │   └── reviewer.py          # 审查 Agent
│   ├── cli/                     # 命令行界面
│   │   ├── __init__.py
│   │   └── main.py              # CLI 主程序
├── workspace/                   # 工作区
│   └── office/                  # Agent 沙箱目录
├── tests_swarmmind/             # 测试套件
├── setup.py                     # 安装配置
├── requirements.txt             # 依赖列表
├── .env.example                 # 环境变量示例
└── README.md                    # 本文档
```

---

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd E:\agent\CyberClaw-main\Swarmmind

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制示例配置
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
```

`.env` 配置示例：

```env
# 默认 LLM 提供商
DEFAULT_PROVIDER=openai
DEFAULT_MODEL=gpt-4o-mini

# OpenAI 兼容 API
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://api.openai.com/v1

# Anthropic (可选)
ANTHROPIC_API_KEY=sk-ant-xxx

# Ollama 本地模型 (可选)
OLLAMA_BASE_URL=http://localhost:11434
```

### 3. 运行

```bash
# 命令行交互模式
python -m swarmmind.cli.main

# 或安装后使用
pip install -e .
swarmmind
```

### 4. 编程接口

```python
import asyncio
from swarmmind.core import SafeOrchestrator

async def main():
    # 创建编排器（启用所有增强功能）
    orchestrator = SafeOrchestrator(
        provider_name="openai",
        model_name="gpt-4o-mini",
        enable_parallel=True,           # 并行执行
        enable_experience=True,         # 经验回放
        enable_anomaly_detection=True   # 异常检测
    )

    # 执行任务
    result = await orchestrator.run("帮我创建一个 hello.txt 文件")
    print(result)

asyncio.run(main())
```

---

## 核心组件

### 多 Agent 协作流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Planner   │────>│   Executor  │────>│  Reviewer   │
│  (规划者)   │     │  (执行者)   │     │  (审查者)   │
└─────────────┘     └─────────────┘     └─────────────┘
      │                   │                   │
      v                   v                   v
  分析意图            执行任务            审查结果
  制定计划            调用工具            质量检查
  read_only          standard           read_only
```

### 权限等级

| 等级 | 描述 | 允许的操作 |
|------|------|-----------|
| `read_only` | 只读权限 | 列文件、读文件、获取时间 |
| `standard` | 标准权限 | 读写文件、执行命令、API调用 |
| `admin` | 管理员权限 | 所有操作 |

### 安全机制

**多层安全防护体系**：

1. **路径安全**：`realpath()` 解析符号链接 + `abspath()` 规范化路径，防止符号链接穿越攻击
2. **命令执行安全**：使用 `shlex.split()` 解析 + `shell=False` + 命令白名单，防止 Shell 注入
3. **代码执行安全**：基于 AST 的代码分析，防止字符串拼接绕过危险函数检测
4. **计算器安全**：基于 AST 的纯数学表达式求值，替代不安全的 `eval()`

**高风险操作确认**：
- Shell 命令执行
- 代码执行
- 文件覆盖写入

---

## 工具列表

### 文件操作

| 工具 | 描述 | 权限 | 风险 |
|------|------|------|------|
| `list_workspace_files` | 列出目录文件 | read_only | low |
| `read_workspace_file` | 读取文件内容 | read_only | low |
| `write_workspace_file` | 写入文件 | standard | medium |
| `execute_workspace_shell` | 执行白名单内 Shell 命令 | standard | high |

### 代码执行

| 工具 | 描述 | 权限 | 风险 |
|------|------|------|------|
| `execute_code` | 执行 Python/JS 代码（AST 安全检查） | standard | high |
| `run_python_script` | 运行工作区内 Python 脚本 | standard | high |

### API 调用

| 工具 | 描述 | 权限 | 风险 |
|------|------|------|------|
| `call_api` | 调用 REST API | standard | medium |
| `call_api_with_auth` | 带认证的 API 调用 | standard | high |
| `test_api_connection` | 测试 API 连接 | read_only | low |

### 内置工具

| 工具 | 描述 |
|------|------|
| `get_current_time` | 获取当前时间 |
| `get_system_model_info` | 获取模型信息 |
| `calculator` | 安全数学计算器（AST 解析） |

---

## 高级功能

### 向量语义记忆

```python
from swarmmind.core import VectorMemoryStore

# 创建向量存储
store = VectorMemoryStore("./memory/vectors")

# 存储记忆
await store.store("用户喜欢使用 Python 进行数据分析", {"type": "preference"})

# 语义检索
results = await store.search("编程语言偏好", top_k=5)
```

### 经验回放

```python
from swarmmind.core import ExperienceStore

store = ExperienceStore("./memory/experience")

# 记录执行经验
await store.record(
    task="创建配置文件",
    plan={"actions": [...]},
    result="成功创建 config.yaml",
    review_passed=True
)

# 检索相似任务经验
experiences = await store.recall("创建配置文件", success_only=True)
```

### 异常行为检测

```python
from swarmmind.core import BehaviorMonitor, check_before_action

# 检查行动前异常
report = check_before_action(
    agent="executor",
    tool="execute_workspace_shell",
    args={"command": "rm -rf /"}
)

if report.is_anomaly:
    print(f"检测到异常: {report.description}")
```

---

## 支持的 LLM 提供商

| 提供商 | 配置值 | 模型示例 |
|--------|--------|----------|
| OpenAI | `openai` | gpt-4o, gpt-4o-mini |
| Anthropic | `anthropic` | claude-3-5-sonnet |
| 阿里云 | `aliyun` | qwen-turbo, qwen-plus |
| 腾讯 | `tencent` | hunyuan-lite |
| 智谱 | `z.ai` | glm-4, glm-4-flash |
| Ollama | `ollama` | llama3, qwen2 |

---

## 测试

```bash
# 运行所有测试
pytest tests_swarmmind/ -v

# 运行特定测试
pytest tests_swarmmind/test_security.py -v
pytest tests_swarmmind/test_sandbox.py -v
```

---

## 依赖项

```
# 核心
langchain>=0.3.0
langgraph>=0.2.0
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0

# 向量存储
chromadb>=0.4.0

# HTTP 客户端
httpx>=0.25.0

# 配置解析
python-dotenv>=1.0.0

# CLI
typer>=0.9.0
rich>=13.0.0
questionary>=2.0.0

# 数据处理
pydantic>=2.0.0
aiosqlite>=0.19.0
```

---

## 安全声明

SwarmMind 采用多层安全架构：

1. **代码执行安全**：基于 AST 的静态分析，防止沙箱逃逸和代码注入
2. **文件系统安全**：符号链接解析 + 路径规范化，防止目录穿越
3. **命令执行安全**：命令白名单 + `shell=False`，防止 Shell 注入
4. **审计日志**：线程安全的异步日志，记录所有 Agent 行为

⚠️ **重要提示**：

1. SwarmMind 设计用于技术研究和受控环境
2. 沙箱机制提供多层防护，但不保证绝对安全
3. 生产环境使用前请进行充分的安全评估
4. 建议在隔离环境中运行不可信任务

---

## 许可证

MIT License

---

## 贡献

欢迎提交 Issue 和 Pull Request！

---

<div align="center">

**SwarmMind** - 让多智能体协作更安全、更智能

</div>
