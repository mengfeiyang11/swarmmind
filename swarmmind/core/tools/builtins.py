"""
SwarmMind 内置工具集
"""

from datetime import datetime
from .base import swarmmind_tool
from .sandbox import (
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
    execute_workspace_shell
)
from .code_sandbox import execute_code, run_python_script
from .api_tool import call_api, call_api_with_auth, test_api_connection
import os


@swarmmind_tool
def get_system_model_info() -> str:
    """
    获取当前 SwarmMind 正在运行的底层大模型信息。
    当用户询问"你是基于什么模型"、"底层大模型是什么"时调用。
    """
    provider = os.getenv("DEFAULT_PROVIDER", "unknown")
    model = os.getenv("DEFAULT_MODEL", "unknown")

    if provider == "unknown" or model == "unknown":
        return "无法获取当前的系统模型配置，可能是环境变量未正确加载。"

    return f"当前使用的模型提供商是: {provider}，具体型号是: {model}。"


@swarmmind_tool
def get_current_time() -> str:
    """获取当前的系统时间和日期。"""
    now = datetime.now()
    return f"当前本地系统时间是: {now.strftime('%Y-%m-%d %H:%M:%S')}"


@swarmmind_tool
def calculator(expression: str) -> str:
    """
    一个安全的数学计算器。
    用于计算基础的数学表达式，例如: '3 * 5' 或 '100 / 4'。

    ⚠️ 安全说明：使用 AST 解析，仅允许数学运算，禁止任何函数调用。
    """
    import ast
    import operator

    # 允许的运算符映射
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # 允许的节点类型
    ALLOWED_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,  # Python 3.8+
        ast.operator,
        ast.unaryop,
    )

    # 向后兼容：Python 3.7 及更早版本使用 ast.Num
    try:
        ALLOWED_NODES = ALLOWED_NODES + (ast.Num,)
    except AttributeError:
        pass

    def _eval_node(node):
        """安全地评估 AST 节点"""
        # 处理 ast.Num (Python < 3.8)
        if hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"不支持的常量类型: {type(node.value)}")
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return ALLOWED_OPERATORS[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            return ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    try:
        # 解析表达式为 AST
        tree = ast.parse(expression, mode='eval')

        # 验证 AST 节点安全性
        for node in ast.walk(tree):
            if not isinstance(node, ALLOWED_NODES):
                raise ValueError(f"安全拦截: 不支持的语法 '{type(node).__name__}'")

        # 安全评估
        result = _eval_node(tree.body)
        return f"表达式 '{expression}' 的计算结果是: {result}"

    except ValueError as e:
        return f"计算出错: {str(e)}"
    except SyntaxError:
        return f"计算出错: 表达式语法无效"
    except ZeroDivisionError:
        return f"计算出错: 除数不能为零"
    except Exception as e:
        return f"计算出错: {str(e)}"


# 内置工具列表
BUILTIN_TOOLS = [
    get_current_time,
    calculator,
    get_system_model_info,
    list_workspace_files,
    read_workspace_file,
    write_workspace_file,
    execute_workspace_shell,
    execute_code,
    run_python_script,
    call_api,
    call_api_with_auth,
    test_api_connection,
]
