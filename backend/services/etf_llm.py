"""ETF 大模型评分：用 tiiuae/Falcon-H1R-7B 根据五年表现输出 0-100 分及理由。"""
from __future__ import annotations

import os
import re
from typing import Any, Optional

# socks 代理会导致 transformers/huggingface_hub 报 Unknown scheme，加载模型时暂时取消代理
_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

from .data import fetch_ohlcv
from backtest.engine import run_backtest, STRATEGY_BUY_HOLD

# 懒加载的 pipeline，首次调用时加载
_llm_pipeline = None
_llm_loaded = False
_llm_error: Optional[str] = None

# 默认从 Hugging Face 拉取；若网络不可达，可先离线下载再设置环境变量走本地
MODEL_ID = "tiiuae/Falcon-H1R-7B"
# 本地模型目录：设置 FALCON_H1R_7B_PATH=/path/to/Falcon-H1R-7B 则不再访问网络（目录内需含 config.json、*.safetensors 等）
_LOCAL_MODEL_ENV = "FALCON_H1R_7B_PATH"
MAX_NEW_TOKENS = 256
INFERENCE_TIMEOUT = 120


def _get_model_path() -> str:
    """优先使用本地路径（避免 Network unreachable），否则用 MODEL_ID。"""
    local = os.environ.get(_LOCAL_MODEL_ENV, "").strip()
    if local and os.path.isdir(local):
        return local
    return MODEL_ID


def _load_pipeline():
    global _llm_pipeline, _llm_loaded, _llm_error
    if _llm_loaded:
        return _llm_pipeline, _llm_error
    _llm_loaded = True
    try:
        import regex  # transformers 依赖，需先安装
    except ImportError:
        _llm_error = (
            "缺少依赖 regex。请在启动后端的同一环境中执行: pip install regex 或 pip install -r requirements.txt"
        )
        return _llm_pipeline, _llm_error
    model_path = _get_model_path()
    saved_proxy = {k: os.environ.pop(k, None) for k in _PROXY_KEYS}
    try:
        from transformers import pipeline as hf_pipeline
        _llm_pipeline = hf_pipeline(
            "text-generation",
            model=model_path,
            trust_remote_code=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )
    except Exception as e:
        err_msg = str(e)
        if "GGUF_CONFIG_MAPPING" in err_msg or "transformers.integrations" in err_msg:
            _llm_error = (
                "transformers 与 accelerate 版本不兼容。请升级：pip install -U 'transformers>=4.46.0' 'accelerate>=0.33.0'"
            )
        elif any(x in err_msg for x in ("Network is unreachable", "Errno 101", "client has been closed")):
            _llm_error = (
                "无法访问 Hugging Face 或连接已关闭。请先离线下载模型并设置环境变量："
                " export FALCON_H1R_7B_PATH=/path/to/Falcon-H1R-7B （目录内需含 config.json、*.safetensors 等）"
            )
        else:
            _llm_error = err_msg
        _llm_pipeline = None
    finally:
        for k, v in saved_proxy.items():
            if v is not None:
                os.environ[k] = v
    return _llm_pipeline, _llm_error


def get_etf_5y_summary(symbol: str, market: str = "us", initial_cash: float = 100_000.0) -> dict[str, Any]:
    """获取单只 ETF 过去 5 年的买入持有表现摘要，供大模型输入。"""
    symbol = (symbol or "").strip().upper()
    if not symbol:
        raise ValueError("标的代码不能为空")
    df = fetch_ohlcv(symbol, period="5y", market=market or "us")
    if df is None or df.empty:
        raise ValueError(f"无法获取 {symbol} 的五年数据")
    res = run_backtest(df, STRATEGY_BUY_HOLD, initial_cash=initial_cash)
    eq = res.equity_curve
    if eq is None or len(eq) == 0:
        raise ValueError(f"{symbol} 无有效净值数据")
    return {
        "symbol": symbol,
        "start_date": str(df.index.min())[:10],
        "end_date": str(df.index.max())[:10],
        "total_return_pct": round(res.total_return * 100, 2),
        "annual_return_pct": round(res.annual_return * 100, 2),
        "volatility_pct": round(res.volatility * 100, 2),
        "sharpe_ratio": round(res.sharpe_ratio, 4),
        "max_drawdown_pct": round(res.max_drawdown * 100, 2),
        "final_value": round(res.final_value, 2),
    }


def _parse_score_and_reason(text: str) -> tuple[int, str]:
    """从模型输出中解析 SCORE: 数字 和 REASON: 文本。"""
    score = 50
    reason = ""
    # SCORE: 75 或 score: 75
    m = re.search(r"SCORE\s*:\s*(\d+)", text, re.I)
    if m:
        score = max(0, min(100, int(m.group(1))))
    m = re.search(r"REASON\s*:\s*(.+?)(?:\n|$)", text, re.S | re.I)
    if m:
        reason = m.group(1).strip()
    if not reason and "reason" in text.lower():
        m = re.search(r"reason\s*:\s*(.+?)(?:\n|$)", text, re.S | re.I)
        if m:
            reason = m.group(1).strip()
    if not reason:
        reason = text.strip()[:500] if text else "无"
    return score, reason


def run_llm_score(symbol: str, summary: dict[str, Any]) -> dict[str, Any]:
    """
    用 Falcon-H1R-7B 根据 summary 生成 0-100 分与理由。
    返回 {"score": int, "reason": str}，失败则抛出或返回默认。
    """
    pipe, err = _load_pipeline()
    if err or pipe is None:
        raise RuntimeError(f"大模型未就绪: {err or '加载失败'}")

    prompt = (
        "You are an ETF analyst. Based on the following 5-year buy-and-hold statistics, "
        "give a single score from 0 to 100 (higher = better long-term hold) and a short reason in English.\n\n"
        f"ETF: {summary.get('symbol', '')}\n"
        f"Period: {summary.get('start_date', '')} to {summary.get('end_date', '')}\n"
        f"Total return: {summary.get('total_return_pct')}%\n"
        f"Annualized return: {summary.get('annual_return_pct')}%\n"
        f"Volatility (annual): {summary.get('volatility_pct')}%\n"
        f"Sharpe ratio: {summary.get('sharpe_ratio')}\n"
        f"Max drawdown: {summary.get('max_drawdown_pct')}%\n\n"
        "Reply with exactly two lines:\nSCORE: <number 0-100>\nREASON: <one or two sentences>"
    )

    try:
        tok = pipe.tokenizer
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token_id = getattr(tok, "eos_token_id", 0)
        out = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    except Exception as e:
        raise RuntimeError(f"模型推理失败: {e}") from e

    if not out or not isinstance(out, list):
        return {"score": 50, "reason": "模型无输出"}
    gen = out[0]
    if isinstance(gen, dict) and "generated_text" in gen:
        text = gen["generated_text"]
    elif isinstance(gen, dict) and "generated_token_ids" in gen:
        text = pipe.tokenizer.decode(gen["generated_token_ids"], skip_special_tokens=True)
    else:
        text = str(gen)
    # 只取模型新生成部分（去掉 prompt）
    if prompt in text:
        text = text.split(prompt, 1)[-1].strip()
    score, reason = _parse_score_and_reason(text)
    return {"score": score, "reason": reason}
