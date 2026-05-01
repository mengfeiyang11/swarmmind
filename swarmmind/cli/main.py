"""
SwarmMind CLI 主程序

提供交互式命令行界面
"""

import asyncio
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from dotenv import load_dotenv

from ..core.orchestrator import SafeOrchestrator
from ..core.config import WORKSPACE_DIR, MEMORY_DIR, WORKSPACE_OFFICE_DIR
from ..core.memory import MemorySystem

load_dotenv()

# 设置控制台编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

console = Console()


def render_welcome():
    """渲染欢迎界面"""
    logo = """
  ____                 __  __      _   _      _
 / ___|_ __ ___  ___  |  \\/  | ___| |_| |__  (_) ___
 \\___ \\ '__/ _ \\/ _ \\ | |\\/| |/ _ \\ __| '_ \\ | |/ _ \\
  ___) | | |  __/  __/ | |  | |  __/ |_| | | |_| |  __/
 |____/|_|  \\___|\\___| |_|  |_|\\___|\\__|_| |_(_|\\___|

    """
    console.print(Panel(
        f"[bold cyan]{logo}[/bold cyan]\n"
        f"[bold]Multi-Agent Collaboration Framework[/bold]\n\n"
        f"[dim]Workspace: {WORKSPACE_DIR}[/dim]\n"
        f"[dim]Office: {WORKSPACE_OFFICE_DIR}[/dim]\n",
        title="SwarmMind",
        border_style="cyan"
    ))


def render_help():
    """渲染帮助信息"""
    console.print(Panel(
        "[bold]Commands:[/bold]\n\n"
        "- Type your question to start\n"
        "- [cyan]exit[/cyan] / [cyan]quit[/cyan] - Exit\n"
        "- [cyan]help[/cyan] - Show help\n"
        "- [cyan]status[/cyan] - Show system status\n"
        "- [cyan]memory[/cyan] - Show user profile\n\n"
        "[bold]Security Tips:[/bold]\n"
        "- File operations limited to office directory\n"
        "- High-risk actions require confirmation\n"
        "- All actions are logged",
        title="Help",
        border_style="blue"
    ))


def render_status():
    """渲染系统状态"""
    provider = os.getenv("DEFAULT_PROVIDER", "not configured")
    model = os.getenv("DEFAULT_MODEL", "not configured")

    console.print(Panel(
        f"[bold]Model Config:[/bold]\n"
        f"  Provider: {provider}\n"
        f"  Model: {model}\n\n"
        f"[bold]Directories:[/bold]\n"
        f"  Workspace: {WORKSPACE_DIR}\n"
        f"  Office: {WORKSPACE_OFFICE_DIR}\n"
        f"  Memory: {MEMORY_DIR}",
        title="System Status",
        border_style="green"
    ))


def render_memory(memory: MemorySystem):
    """渲染用户画像"""
    profile = memory.load_user_profile()

    console.print(Panel(
        profile,
        title="User Profile",
        border_style="purple"
    ))


async def run_streaming_response(orchestrator: SafeOrchestrator, user_input: str):
    """流式输出响应"""
    buffer = ""

    with Live(console=console, refresh_per_second=10) as live:
        async for chunk in orchestrator.stream(user_input):
            buffer += chunk
            live.update(Text(buffer))


async def main_loop():
    """主交互循环"""
    render_welcome()

    # 初始化
    provider = os.getenv("DEFAULT_PROVIDER", "openai")
    model = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    orchestrator = SafeOrchestrator(provider_name=provider, model_name=model)
    memory = MemorySystem(WORKSPACE_DIR)

    console.print("[dim]Tip: Type 'help' for commands[/dim]\n")

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold cyan]Goodbye![/bold cyan]")
                break

            elif user_input.lower() == "help":
                render_help()

            elif user_input.lower() == "status":
                render_status()

            elif user_input.lower() == "memory":
                render_memory(memory)

            else:
                try:
                    await run_streaming_response(orchestrator, user_input)
                except Exception as e:
                    console.print(f"[bold red]Error: {e}[/bold red]")

        except KeyboardInterrupt:
            console.print("\n[bold cyan]Goodbye![/bold cyan]")
            break
        except EOFError:
            break


def run():
    """启动 CLI"""
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import threading
            error = [None]

            def _run_in_thread():
                try:
                    asyncio.run(main_loop())
                except Exception as e:
                    error[0] = e

            t = threading.Thread(target=_run_in_thread)
            t.start()
            t.join()

            if error[0]:
                raise error[0]
        else:
            asyncio.run(main_loop())
    except KeyboardInterrupt:
        console.print("[bold cyan]Goodbye![/bold cyan]")
    except Exception as e:
        print(f"Startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
