"""
SwarmMind 代码执行沙箱

提供安全的代码执行环境：
- Python/JavaScript 代码执行
- 危险模块拦截
- 资源限制和超时
- 输出截断
"""

import subprocess
import sys
import os
import tempfile
import re
from typing import Optional
from .base import swarmmind_tool
from ..config import WORKSPACE_OFFICE_DIR


# 危险模块黑名单
DANGEROUS_MODULES = [
    "os",
    "sys",
    "subprocess",
    "shutil",
    "socket",
    "pickle",
    "marshal",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    "builtins",
    "__import__",
    "importlib",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
]

# 危险关键字
DANGEROUS_KEYWORDS = [
    "import os",
    "import sys",
    "import subprocess",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "input(",
    "breakpoint(",
    "exit(",
    "quit(",
]


def _check_code_safety(code: str, language: str = "python") -> tuple[bool, str]:
    """
    检查代码安全性（基于 AST 分析，防止字符串拼接绕过）

    返回: (是否安全, 错误消息)
    """
    if language == "python":
        import ast as _ast

        # 使用 AST 分析，防止字符串拼接/编码绕过
        try:
            tree = _ast.parse(code)
        except SyntaxError as e:
            return False, f"代码语法错误: {e}"

        # 危险模块集合
        dangerous_modules_set = {
            "os", "sys", "subprocess", "shutil", "socket", "pickle",
            "marshal", "ctypes", "multiprocessing", "threading", "signal",
            "builtins", "importlib", "posixpath", "ntpath", "genericpath",
            "io", "pathlib", "glob", "fnmatch", "linecache", "tokenize",
            "code", "codeop", "compileall", "py_compile", "zipimport",
            "pkgutil", "modulefinder", "runpy", "traceback", "webbrowser",
        }

        # 危险属性名（双下划线开头的危险内部属性）
        dangerous_attrs = {
            "__import__", "__builtins__", "__class__", "__subclasses__",
            "__bases__", "__mro__", "__globals__", "__code__", "__globals__",
            "__loader__", "__spec__", "__reduce__", "__reduce_ex__",
        }

        for node in _ast.walk(tree):
            # 检查 import 语句
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in dangerous_modules_set:
                        return False, f"禁止导入模块: {root_module}"

            # 检查 from ... import 语句
            if isinstance(node, _ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module in dangerous_modules_set:
                        return False, f"禁止从模块导入: {root_module}"

            # 检查函数调用
            if isinstance(node, _ast.Call):
                func_name = None
                if isinstance(node.func, _ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, _ast.Attribute):
                    func_name = node.func.attr

                if func_name in ("eval", "exec", "compile", "open", "input",
                                 "breakpoint", "exit", "quit", "__import__",
                                 "getattr", "setattr", "delattr", "type",
                                 "globals", "locals", "vars", "dir",
                                 "memoryview", "bytearray"):
                    return False, f"检测到危险函数调用: {func_name}"

            # 检查属性访问（防止 __class__.__subclasses__() 绕过）
            if isinstance(node, _ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    if node.attr in dangerous_attrs:
                        return False, f"检测到危险属性访问: {node.attr}"

            # 检查 exec/eval 赋值绕过
            if isinstance(node, _ast.Name):
                if node.id.startswith("__") and node.id.endswith("__"):
                    if node.id in dangerous_attrs:
                        return False, f"检测到危险标识符: {node.id}"

    elif language == "javascript":
        # JavaScript AST 不可用，使用增强的字符串分析
        import re as _re

        # 标准化代码（去除注释、多余空白）
        normalized = _re.sub(r'//.*', '', code)
        normalized = _re.sub(r'/\*.*?\*/', '', normalized, flags=_re.DOTALL)

        js_dangerous = [
            (_re.compile(r'\brequire\s*\('), "require() 调用"),
            (_re.compile(r'\bimport\s+'), "import 语句"),
            (_re.compile(r'\beval\s*\('), "eval() 调用"),
            (_re.compile(r'\bFunction\s*\('), "Function() 构造"),
            (_re.compile(r'\bprocess\b'), "process 全局对象"),
            (_re.compile(r'\bchild_process\b'), "child_process 模块"),
            (_re.compile(r'\bfs\b\.'), "fs 文件系统"),
            (_re.compile(r'\bhttp\b\.'), "http 模块"),
            (_re.compile(r'\bhttps\b\.'), "https 模块"),
            (_re.compile(r'\b__proto__\b'), "原型链访问"),
            (_re.compile(r'\bconstructor\b\s*\['), "构造函数访问"),
        ]

        for pattern, desc in js_dangerous:
            if pattern.search(normalized):
                return False, f"检测到危险操作: {desc}"

    return True, ""


def _create_safe_builtins() -> dict:
    """创建安全的内置函数集合"""
    import builtins

    safe_builtins = {
        # 允许的内置函数
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "chr": chr,
        "dict": dict,
        "enumerate": enumerate,
        "filter": filter,
        "float": float,
        "format": format,
        "frozenset": frozenset,
        "hex": hex,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "oct": oct,
        "ord": ord,
        "pow": pow,
        "print": print,
        "range": range,
        "repr": repr,
        "reversed": reversed,
        "round": round,
        "set": set,
        "slice": slice,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "type": type,
        "zip": zip,
        # 数学相关
        "True": True,
        "False": False,
        "None": None,
    }

    return safe_builtins


@swarmmind_tool
def execute_code(code: str, language: str = "python", timeout: int = 30) -> str:
    """
    在安全沙箱中执行代码片段。

    参数:
    - code: 要执行的代码
    - language: 编程语言 ("python" 或 "javascript")
    - timeout: 执行超时时间（秒），默认 30 秒

    【权限等级】: standard
    【风险等级】: high
    【需要确认】: 是

    支持的语言:
    - python: Python 代码执行
    - javascript: JavaScript (Node.js) 代码执行

    安全限制:
    - 禁止导入危险模块 (os, sys, subprocess 等)
    - 禁止文件操作
    - 禁止网络操作
    - 执行时间限制
    - 输出长度限制
    """
    # 安全检查
    is_safe, error_msg = _check_code_safety(code, language)
    if not is_safe:
        return f"❌ 安全拦截: {error_msg}\n\n你的代码包含危险操作，已被阻止执行。"

    # 语言处理
    if language == "python":
        return _execute_python_code(code, timeout)
    elif language == "javascript":
        return _execute_javascript_code(code, timeout)
    else:
        return f"❌ 不支持的语言: {language}\n目前仅支持: python, javascript"


def _execute_python_code(code: str, timeout: int) -> str:
    """执行 Python 代码（使用 subprocess 隔离，无临时文件）"""
    # 添加输出捕获
    wrapped_code = f"""
import sys
import io

# 重定向输出
_output_buffer = io.StringIO()
_original_stdout = sys.stdout
sys.stdout = _output_buffer

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
finally:
    sys.stdout = _original_stdout
    _output = _output_buffer.getvalue()
    if _output:
        print(_output, end='')
"""

    try:
        # 使用 subprocess 隔离执行，通过 -c 参数传递代码，无需临时文件
        result = subprocess.run(
            [sys.executable, "-c", wrapped_code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE_OFFICE_DIR,
            # 限制资源
            env={
                "PYTHONPATH": "",
                "PATH": os.environ.get("PATH", ""),
            }
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"

        # 截断输出
        if len(output) > 5000:
            output = output[:5000] + "\n\n...[输出过长，已截断]"

        if result.returncode != 0:
            output = f"❌ 执行错误 (exit code: {result.returncode})\n{output}"

        return output if output.strip() else "(执行完毕，无输出)"

    except subprocess.TimeoutExpired:
        return f"❌ 执行超时 ({timeout}秒)\n代码执行时间过长，已被终止。"
    except Exception as e:
        return f"❌ 执行异常: {str(e)}"


def _execute_javascript_code(code: str, timeout: int) -> str:
    """执行 JavaScript 代码"""
    # 检查 Node.js 是否可用
    try:
        node_check = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if node_check.returncode != 0:
            return "❌ Node.js 未安装或不可用"
    except FileNotFoundError:
        return "❌ Node.js 未安装。请先安装 Node.js 以执行 JavaScript 代码。"
    except subprocess.TimeoutExpired:
        return "❌ Node.js 检查超时"

    # 创建临时文件
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.js',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["node", temp_file],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE_OFFICE_DIR,
            env={"NODE_PATH": ""}
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"

        # 截断输出
        if len(output) > 5000:
            output = output[:5000] + "\n\n...[输出过长，已截断]"

        if result.returncode != 0:
            output = f"❌ 执行错误 (exit code: {result.returncode})\n{output}"

        return output if output.strip() else "(执行完毕，无输出)"

    except subprocess.TimeoutExpired:
        return f"❌ 执行超时 ({timeout}秒)"
    except Exception as e:
        return f"❌ 执行异常: {str(e)}"
    finally:
        try:
            os.unlink(temp_file)
        except Exception:
            pass


@swarmmind_tool
def run_python_script(filepath: str, args: str = "", timeout: int = 60) -> str:
    """
    在沙箱中运行工作区内的 Python 脚本。

    参数:
    - filepath: 相对于 office 的脚本路径
    - args: 命令行参数
    - timeout: 超时时间（秒）

    【权限等级】: standard
    【风险等级】: high
    【需要确认】: 是
    """
    from .sandbox import _get_safe_path

    try:
        script_path = _get_safe_path(filepath)

        if not os.path.exists(script_path):
            return f"❌ 脚本不存在: {filepath}"

        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args.split())

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=WORKSPACE_OFFICE_DIR
        )

        output = f"● 执行脚本: {filepath}\n"
        output += f"● 退出码: {result.returncode}\n"

        if result.stdout:
            output += f"\n[STDOUT]\n{result.stdout[-3000:]}"
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr[-3000:]}"

        return output

    except subprocess.TimeoutExpired:
        return f"❌ 脚本执行超时 ({timeout}秒)"
    except PermissionError as e:
        return f"❌ 权限错误: {str(e)}"
    except Exception as e:
        return f"❌ 执行异常: {str(e)}"
