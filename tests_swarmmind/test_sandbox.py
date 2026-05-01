"""
SwarmMind 沙盒工具安全测试
"""

import pytest

# 注意：这些测试需要在隔离环境中运行


class TestSandboxPathSafety:
    """沙盒路径安全测试"""

    def test_safe_path_within_workspace(self):
        """测试合法路径"""
        from swarmmind.core.tools.sandbox import _get_safe_path
        from swarmmind.core.config import WORKSPACE_OFFICE_DIR

        # 合法路径
        safe_path = _get_safe_path("test.txt")
        assert safe_path.startswith(WORKSPACE_OFFICE_DIR)

        safe_path = _get_safe_path("subdir/test.txt")
        assert safe_path.startswith(WORKSPACE_OFFICE_DIR)

    def test_path_traversal_blocked(self):
        """测试路径遍历攻击被拦截"""
        from swarmmind.core.tools.sandbox import _get_safe_path

        # 相对路径越权
        with pytest.raises(PermissionError):
            _get_safe_path("../../../etc/passwd")

        with pytest.raises(PermissionError):
            _get_safe_path("../../windows/system32")

    def test_absolute_path_blocked(self):
        """测试绝对路径被拦截"""
        from swarmmind.core.tools.sandbox import _get_safe_path

        # Unix 绝对路径
        with pytest.raises(PermissionError):
            _get_safe_path("/etc/passwd")

        # Windows 盘符路径 (在 Windows 上测试)
        # 注意：这个测试在 Linux 上可能不会触发


class TestSandboxShellSafety:
    """沙盒 Shell 安全测试"""

    def test_dangerous_command_blocked(self):
        """测试危险命令被拦截"""
        from swarmmind.core.tools.sandbox import execute_workspace_shell

        # 相对路径越权
        result = execute_workspace_shell.invoke({"command": "cat ../../../etc/passwd"})
        assert "权限拒绝" in result or "危险" in result

        # Unix 绝对路径
        result = execute_workspace_shell.invoke({"command": "cat /etc/passwd"})
        assert "权限拒绝" in result or "危险" in result

    def test_safe_command_allowed(self):
        """测试安全命令可以执行"""
        from swarmmind.core.tools.sandbox import execute_workspace_shell

        # 简单的列出文件命令
        result = execute_workspace_shell.invoke({"command": "echo 'hello'"})
        assert "执行命令" in result
