"""
SwarmMind 沙盒安全工具

提供安全的文件操作和 Shell 执行功能。
所有操作被限制在 workspace/office 目录内。
"""

import os
import subprocess
import shlex
import platform
from .base import swarmmind_tool
from ..config import WORKSPACE_OFFICE_DIR

SYS_OS = platform.system()


def _get_safe_path(relative_path: str) -> str:
    """
    将相对路径转换为绝对路径，并检查是否越界。
    如果尝试访问沙盒外的路径，直接拦截。

    安全措施：
    - 使用 realpath() 解析符号链接，防止符号链接穿越
    - 验证解析后的路径仍在沙盒内
    """
    base_dir = os.path.abspath(WORKSPACE_OFFICE_DIR)
    # 先解析为绝对路径（不含 .. 等）
    intermediate = os.path.join(base_dir, relative_path)
    # 关键：使用 realpath() 解析符号链接
    target_path = os.path.realpath(intermediate)

    # 核心防御：目标路径必须以 WORKSPACE_OFFICE_DIR 开头
    if not target_path.startswith(base_dir):
        raise PermissionError(
            f"越权拦截：你试图访问沙盒外的路径 '{relative_path}'！"
            "你只能在 office 工位内活动。"
        )

    return target_path


@swarmmind_tool
def list_workspace_files(sub_dir: str = "") -> str:
    """
    查看 office 工位里有哪些文件和文件夹。
    如果 sub_dir 为空，则查看工位根目录。

    【权限等级】: read_only
    【风险等级】: low
    """
    try:
        target_dir = _get_safe_path(sub_dir)
        if not os.path.exists(target_dir):
            return f"目录不存在：{sub_dir}"

        items = os.listdir(target_dir)
        if not items:
            return f"[{sub_dir if sub_dir else 'office 根目录'}] 是空的。"

        result = []
        for item in items:
            item_path = os.path.join(target_dir, item)
            item_type = "📁" if os.path.isdir(item_path) else "📄"
            result.append(f"{item_type} {item}")

        return "\n".join(result)
    except Exception as e:
        return str(e)


@swarmmind_tool
def read_workspace_file(filepath: str) -> str:
    """
    读取 office 工位里指定文件的内容。
    filepath 参数应该是相对于 office 的路径。

    【权限等级】: read_only
    【风险等级】: low
    """
    try:
        target_path = _get_safe_path(filepath)
        if not os.path.exists(target_path):
            return f"文件不存在：{filepath}"

        with open(target_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 防爆截断：防止读取超大文件撑爆 Token
        if len(content) > 10000:
            return content[:10000] + "\n\n...[内容过长，已被安全截断]..."
        return content
    except Exception as e:
        return str(e)


@swarmmind_tool
def write_workspace_file(filepath: str, content: str, mode: str = "w") -> str:
    """
    在 office 工位里操作文件内容。

    参数说明:
    - filepath: 相对路径
    - content: 要写入的内容
    - mode: 写入模式 ("w" 覆盖/新建, "a" 追加)

    【权限等级】: standard
    【风险等级】: medium
    【需要确认】: 覆盖模式时建议确认

    ⚠️ 安全规范：
    1. 禁止编写跳出 office 工位相关的任何脚本
    2. 修改长文件时建议先读取，替换后再重写
    """
    try:
        target_path = _get_safe_path(filepath)

        if mode not in ["w", "a"]:
            return "❌ 错误：mode 参数必须是 'w' (覆盖) 或 'a' (追加)。"

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        with open(target_path, mode, encoding="utf-8") as f:
            if mode == "a" and not content.startswith("\n"):
                f.write("\n" + content)
            else:
                f.write(content)

        action = "覆盖/新建" if mode == "w" else "追加"
        return f"● 成功以 {action} 模式写入文件：{filepath} (共 {len(content)} 字符)"
    except Exception as e:
        return str(e)


@swarmmind_tool
def execute_workspace_shell(command: str) -> str:
    """
    在 office 工位中执行 Shell 命令。

    【权限等级】: standard
    【风险等级】: high
    【需要确认】: 必须用户确认

    ⚠️ 【极其重要的环境限制】：
    1. 跨平台注意：根据系统使用对应命令（Win 用 dir/del，Linux 用 ls/rm）
    2. 非交互式终端！所有命令必须携带免确认参数
    3. 禁止使用 cd 跳出当前目录
    4. 每次执行都是独立进程，需要进入子目录请用相对路径
    5. 禁止一切形式跳出 office 工位
    """
    try:
        import shlex

        # 安全重写：使用 shlex.split 解析命令，避免 shell=True
        # 这样可以防止 shell 元字符（;, |, &, $, ` 等）被解释执行
        try:
            command_parts = shlex.split(command)
        except ValueError as e:
            return f"❌ 命令解析错误: {str(e)}"

        if not command_parts:
            return "❌ 错误：命令为空"

        # 验证可执行文件路径（防止相对路径和绝对路径逃逸）
        executable = command_parts[0]

        # 1. 不允许绝对路径或包含 .. 的可执行文件
        if os.path.isabs(executable):
            return "❌ 权限拒绝：禁止使用绝对路径执行程序"

        if ".." in executable or executable.startswith("~"):
            return "❌ 权限拒绝：检测到危险的目录跳转"

        # 2. 检查是否为常见的安全内置命令（允许在白名单内）
        # 这些命令是系统内置的，无需完整路径
        ALLOWED_COMMANDS = {
            # Windows
            "dir", "del", "copy", "type", "echo", "cd", "md", "mkdir",
            "rd", "rmdir", "ren", "move", "cls", "date", "time", "cls",
            "where", "whoami", "hostname", "set", "pause",
            # Unix/Linux
            "ls", "cat", "cp", "mv", "rm", "mkdir", "rmdir", "pwd",
            "echo", "date", "whoami", "hostname", "env", "printenv",
            "head", "tail", "grep", "find", "wc", "sort", "uniq",
            "tar", "gzip", "gunzip", "zip", "unzip", "curl", "wget",
            "python", "python3", "node", "npm", "pip", "pip3",
        }

        if executable not in ALLOWED_COMMANDS:
            return (f"❌ 权限拒绝：命令 '{executable}' 不在白名单中。\n"
                    f"可用的安全命令包括: {', '.join(sorted(ALLOWED_COMMANDS))}")

        # 使用 subprocess 直接执行，不使用 shell
        result = subprocess.run(
            command_parts,
            shell=False,  # 关键：不使用 shell 解析
            cwd=WORKSPACE_OFFICE_DIR,
            capture_output=True,
            encoding='utf-8',
            errors='replace',
            timeout=60
        )

        output = f"● 当前系统: {SYS_OS}\n"
        output += f"● 执行命令: `{command}`\n"
        output += f"● 退出码: {result.returncode}\n"

        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""

        if result.returncode != 0 and ("prompt" in stderr.lower() or "y/n" in stdout.lower()):
            output += "\n💡 系统提示：命令可能由于交互式等待而失败。请重试并添加 -y 参数！"

        if stdout:
            output += f"\n[STDOUT]\n{stdout[-2000:] if len(stdout) > 2000 else stdout}"
        if stderr:
            output += f"\n[STDERR]\n{stderr[-2000:] if len(stderr) > 2000 else stderr}"

        if not stdout and not stderr:
            if result.returncode == 0:
                output += "\n(静默执行完毕：无终端输出)"
            else:
                output += "\n(异常退出：Exit Code 非 0，无错误日志输出)"

        return output

    except subprocess.TimeoutExpired:
        return "❌ 严重错误：命令执行超时（60s）被熔断！请检查是否有阻塞式交互。"
    except FileNotFoundError:
        return f"❌ 命令未找到：'{command_parts[0]}' 不是有效的命令或不在 PATH 中"
    except Exception as e:
        return f"❌ 执行异常：{str(e)}"