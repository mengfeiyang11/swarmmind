"""
SwarmMind LLM 提供商适配器

支持: OpenAI / Anthropic / 阿里云 / 腾讯 / Z.AI / Ollama
"""

import os
from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from dotenv import load_dotenv

load_dotenv()

# 各厂商 OpenAI 兼容接口地址
COMPATIBLE_BASE_URLS = {
    "aliyun": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "z.ai": "https://open.bigmodel.cn/api/paas/v4",
    "tencent": "https://api.hunyuan.cloud.tencent.com/v1"
}


def get_provider(
    provider_name: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any
) -> BaseChatModel:
    """模型适配器工厂"""
    provider_name = provider_name.lower()

    if provider_name in ["openai", "aliyun", "dashscope", "z.ai", "tencent", "other"]:
        from langchain_openai import ChatOpenAI

        current_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not current_api_key:
            raise ValueError("未找到 API Key！请确保 .env 中配置了 OPENAI_API_KEY")

        final_base_url = base_url or os.environ.get("OPENAI_API_BASE")
        if not final_base_url:
            final_base_url = COMPATIBLE_BASE_URLS.get(provider_name)

        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )

    elif provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic

        current_api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not current_api_key:
            raise ValueError("未找到 ANTHROPIC_API_KEY 环境变量！")

        final_base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")

        return ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            api_key=current_api_key,
            base_url=final_base_url,
            **kwargs
        )

    elif provider_name == "ollama":
        from langchain_community.chat_models import ChatOllama

        final_base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url=final_base_url,
            **kwargs
        )

    else:
        raise ValueError(f"不支持的模型提供商: {provider_name}")
