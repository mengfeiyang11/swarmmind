"""
SwarmMind API 调用工具

提供标准化的 REST API 调用能力：
- 支持 GET/POST/PUT/DELETE 方法
- 请求模板系统
- 安全限制和超时
"""

import json
import re
import ipaddress
import socket
from typing import Dict, Any, Optional, List
from .base import swarmmind_tool


def _is_private_url(url: str) -> bool:
    """检查是否为内网地址（增强版：覆盖 IPv4/IPv6/特殊表示）"""
    # 提取主机名
    host_match = re.search(r"https?://([^/:]+)", url)
    if not host_match:
        return False

    host = host_match.group(1)

    # 去掉 IPv6 方括号
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]

    # 检查 localhost 变体
    if host.lower() in ("localhost", "localhost.localdomain"):
        return True

    # 尝试直接解析为 IP 地址
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
        # IPv4-mapped IPv6 (如 ::ffff:127.0.0.1)
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
            if ip.ipv4_mapped.is_private or ip.ipv4_mapped.is_loopback:
                return True
        return False
    except ValueError:
        pass

    # 非 IP 格式的主机名，通过 DNS 解析检查
    try:
        resolved = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _, _, _, sockaddr in resolved:
            addr_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(addr_str)
                if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                    return True
                if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
                    if ip.ipv4_mapped.is_private or ip.ipv4_mapped.is_loopback:
                        return True
            except ValueError:
                continue
    except (socket.gaierror, OSError):
        pass

    return False


def _validate_url(url: str) -> tuple[bool, str]:
    """验证 URL 安全性"""
    # 检查协议
    if not url.startswith(("http://", "https://")):
        return False, "URL 必须以 http:// 或 https:// 开头"

    # 检查内网地址（包括 DNS 解析检查）
    if _is_private_url(url):
        return False, "禁止访问内网地址"

    # 检查危险协议
    dangerous_protocols = ["file://", "ftp://", "javascript:", "data:"]
    for proto in dangerous_protocols:
        if url.lower().startswith(proto):
            return False, f"禁止使用协议: {proto}"

    # 检查用户信息（防止 user:pass@host 格式）
    userinfo_match = re.search(r"https?://([^/@]+)@", url)
    if userinfo_match:
        return False, "URL 中不允许包含用户凭据"

    return True, ""


@swarmmind_tool
def call_api(
    url: str,
    method: str = "GET",
    headers: Dict[str, str] = None,
    body: Dict[str, Any] = None,
    params: Dict[str, str] = None,
    timeout: int = 30
) -> str:
    """
    调用 REST API。

    参数:
    - url: API 地址
    - method: HTTP 方法 (GET, POST, PUT, DELETE)
    - headers: 请求头
    - body: 请求体 (POST/PUT)
    - params: URL 查询参数
    - timeout: 超时时间（秒）

    【权限等级】: standard
    【风险等级】: medium
    【需要确认】: 否

    返回:
    - API 响应内容或错误信息

    示例:
    - call_api("https://api.example.com/data", "GET")
    - call_api("https://api.example.com/users", "POST", body={"name": "test"})
    """
    # 验证 URL
    is_valid, error_msg = _validate_url(url)
    if not is_valid:
        return f"❌ URL 验证失败: {error_msg}"

    # 规范化方法名
    method = method.upper()
    if method not in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
        return f"❌ 不支持的 HTTP 方法: {method}"

    try:
        import httpx
    except ImportError:
        return "❌ httpx 未安装。请运行: pip install httpx"

    # 构建请求
    request_headers = headers or {}
    request_headers.setdefault("User-Agent", "SwarmMind-Agent/1.0")
    request_headers.setdefault("Accept", "application/json")

    # 添加查询参数
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query_string}" if "?" not in url else f"{url}&{query_string}"

    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            request_args = {
                "url": url,
                "method": method,
                "headers": request_headers,
            }

            if body and method in ["POST", "PUT", "PATCH"]:
                # 根据 Content-Type 处理 body
                content_type = request_headers.get("Content-Type", "application/json")
                if "application/json" in content_type:
                    request_args["json"] = body
                else:
                    request_args["data"] = body

            response = client.request(**request_args)

            # 构建输出
            output = f"● 请求: {method} {url}\n"
            output += f"● 状态码: {response.status_code}\n"

            # 解析响应
            try:
                response_data = response.json()
                response_text = json.dumps(response_data, ensure_ascii=False, indent=2)
            except Exception:
                response_text = response.text

            # 截断响应
            if len(response_text) > 3000:
                response_text = response_text[:3000] + "\n\n...[响应过长，已截断]"

            output += f"\n[响应内容]\n{response_text}"

            # 添加响应头信息（如果有用的）
            content_type = response.headers.get("content-type", "")
            if content_type:
                output += f"\n\n[Content-Type: {content_type}]"

            return output

    except httpx.TimeoutException:
        return f"❌ 请求超时 ({timeout}秒)\nURL: {url}"
    except httpx.ConnectError as e:
        return f"❌ 连接失败: {str(e)}\nURL: {url}"
    except Exception as e:
        return f"❌ 请求异常: {str(e)}"


@swarmmind_tool
def call_api_with_auth(
    url: str,
    method: str = "GET",
    auth_type: str = "bearer",
    auth_token: str = "",
    headers: Dict[str, str] = None,
    body: Dict[str, Any] = None,
    timeout: int = 30
) -> str:
    """
    调用需要认证的 REST API。

    参数:
    - url: API 地址
    - method: HTTP 方法
    - auth_type: 认证类型 (bearer, basic, api_key)
    - auth_token: 认证令牌
    - headers: 额外请求头
    - body: 请求体
    - timeout: 超时时间

    【权限等级】: standard
    【风险等级】: high
    【需要确认】: 是
    """
    # 构建认证头
    auth_headers = headers or {}

    if auth_type == "bearer":
        auth_headers["Authorization"] = f"Bearer {auth_token}"
    elif auth_type == "basic":
        import base64
        auth_headers["Authorization"] = f"Basic {base64.b64encode(auth_token.encode()).decode()}"
    elif auth_type == "api_key":
        auth_headers["X-API-Key"] = auth_token
    else:
        return f"❌ 不支持的认证类型: {auth_type}"

    return call_api(
        url=url,
        method=method,
        headers=auth_headers,
        body=body,
        timeout=timeout
    )


@swarmmind_tool
def test_api_connection(url: str, timeout: int = 10) -> str:
    """
    测试 API 连接是否可用。

    参数:
    - url: API 地址
    - timeout: 超时时间

    【权限等级】: read_only
    【风险等级】: low
    """
    # 验证 URL
    is_valid, error_msg = _validate_url(url)
    if not is_valid:
        return f"❌ URL 验证失败: {error_msg}"

    try:
        import httpx
    except ImportError:
        return "❌ httpx 未安装"

    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.options(url)

            output = f"● URL: {url}\n"
            output += f"● 状态: 连接成功\n"
            output += f"● 状态码: {response.status_code}\n"

            # 显示支持的 HTTP 方法
            allow = response.headers.get("Allow", "")
            if allow:
                output += f"● 支持方法: {allow}\n"

            # 显示服务器信息
            server = response.headers.get("Server", "")
            if server:
                output += f"● 服务器: {server}"

            return output

    except httpx.TimeoutException:
        return f"❌ 连接超时 ({timeout}秒)"
    except httpx.ConnectError:
        return "❌ 无法连接到服务器"
    except Exception as e:
        return f"❌ 连接异常: {str(e)}"


# API 模板管理
class APITemplateManager:
    """API 模板管理器"""

    def __init__(self):
        self._templates: Dict[str, Dict[str, Any]] = {}

    def register_template(
        self,
        name: str,
        url_template: str,
        default_method: str = "GET",
        default_headers: Dict[str, str] = None,
        description: str = ""
    ) -> None:
        """注册 API 模板"""
        self._templates[name] = {
            "url_template": url_template,
            "default_method": default_method,
            "default_headers": default_headers or {},
            "description": description
        }

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """获取模板"""
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self._templates.keys())

    def call_template(
        self,
        name: str,
        url_params: Dict[str, str] = None,
        method: str = None,
        headers: Dict[str, str] = None,
        body: Dict[str, Any] = None
    ) -> str:
        """使用模板调用 API"""
        template = self.get_template(name)
        if not template:
            return f"❌ 模板不存在: {name}"

        # 构建 URL
        url = template["url_template"]
        if url_params:
            for key, value in url_params.items():
                url = url.replace(f"{{{key}}}", value)

        # 合并配置
        final_method = method or template["default_method"]
        final_headers = {**template["default_headers"], **(headers or {})}

        return call_api(
            url=url,
            method=final_method,
            headers=final_headers,
            body=body
        )


# 全局模板管理器
api_template_manager = APITemplateManager()
