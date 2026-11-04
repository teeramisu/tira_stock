"""统一 LLM 客户端：优先 DeepSeek API（OpenAI 兼容），回退本地 transformers 模型。

环境变量:
    DEEPSEEK_API_KEY   — DeepSeek API 密钥（必填才能使用 API 模式）
    DEEPSEEK_BASE_URL  — API 基地址，默认 https://api.deepseek.com
    DEEPSEEK_MODEL     — 模型名，默认 deepseek-chat
"""
from __future__ import annotations

import os
from typing import Optional

_client = None
_client_loaded = False
_client_error: Optional[str] = None


def _get_config() -> dict:
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", "").strip(),
        "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip(),
    }


def is_api_available() -> bool:
    """检查 DeepSeek API 是否已配置。"""
    return bool(_get_config()["api_key"])


def get_client():
    """获取 OpenAI 兼容客户端（DeepSeek API）。返回 (client, error)。"""
    global _client, _client_loaded, _client_error
    if _client_loaded:
        return _client, _client_error
    _client_loaded = True

    cfg = _get_config()
    if not cfg["api_key"]:
        _client_error = "未配置 DEEPSEEK_API_KEY。请在 .env 中填写你的 DeepSeek API 密钥"
        return None, _client_error

    try:
        from openai import OpenAI
        # 某些环境把 ALL_PROXY 设成 socks://，httpx 无 socks 支持时会报
        # "Unknown scheme for proxy URL"；临时移除 ALL_PROXY（HTTP(S)_PROXY 仍生效）。
        _saved_proxy = {k: os.environ.pop(k, None) for k in ("ALL_PROXY", "all_proxy")}
        try:
            _client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=60.0)
        finally:
            for k, v in _saved_proxy.items():
                if v is not None:
                    os.environ[k] = v
        _client._model_name = cfg["model"]  # 存到 client 上方便调用
    except ImportError:
        _client_error = "缺少依赖 openai。请执行: pip install openai"
        return None, _client_error
    except Exception as e:
        _client_error = f"初始化 OpenAI 客户端失败: {e}"
        return None, _client_error

    return _client, None


def chat_completion(prompt: str, max_tokens: int = 300, system: str = "") -> str:
    """调用 DeepSeek API 获取文本回复。失败返回空字符串。"""
    client, err = get_client()
    if err or client is None:
        raise RuntimeError(err or "LLM 客户端不可用")

    model = getattr(client, "_model_name", "deepseek-chat")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        raise RuntimeError(f"DeepSeek API 调用失败: {e}") from e
