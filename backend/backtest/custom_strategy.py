"""
自定义 Python 策略执行：在受限环境中运行用户代码，返回持仓序列并跑回测。
约定：用户代码必须定义 signal(df) -> Series|list，返回与 df 同长度的 0/1 序列（1=持多，0=空仓）。
"""
from __future__ import annotations

import io
import multiprocessing
from typing import Any, Optional
import pandas as pd
import numpy as np

from .engine import _run_from_position, BacktestResult, _ensure_ohlcv

# 允许的 builtins，禁止 open/__import__/eval/exec 等
_SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len",
    "list", "map", "max", "min", "range", "round", "set", "sorted", "sum",
    "tuple", "zip", "str", "repr", "isinstance", "filter", "reversed",
    "hasattr", "getattr", "setattr", "TypeError", "ValueError",
}


def _restricted_globals() -> dict:
    import builtins
    g = {k: getattr(builtins, k) for k in _SAFE_BUILTINS if hasattr(builtins, k)}
    g["None"] = None
    g["True"] = True
    g["False"] = False
    out = {"__builtins__": g, "pd": pd, "np": np}
    return out


def _run_user_signal(code: str, df: pd.DataFrame) -> pd.Series:
    """在受限命名空间中执行 code，调用 signal(df) 并返回 position。"""
    df = _ensure_ohlcv(df)
    g = _restricted_globals()
    exec(code, g)
    if "signal" not in g:
        raise ValueError("代码中必须定义函数 signal(df)，返回与 df 同长度的 0/1 序列（1=持多，0=空仓）")
    fn = g["signal"]
    position = fn(df)
    if position is None:
        raise ValueError("signal(df) 不能返回 None")
    if isinstance(position, list):
        position = pd.Series(position, index=df.index)
    if not isinstance(position, pd.Series):
        position = pd.Series(list(position), index=df.index)
    if len(position) != len(df):
        raise ValueError(f"signal(df) 返回长度 {len(position)} 与 df 长度 {len(df)} 不一致")
    return position


def _worker_run_signal(code: str, df_bytes: bytes) -> tuple:
    """执行用户代码并返回 (ok, position_or_error)."""
    try:
        df = pd.read_pickle(io.BytesIO(df_bytes))
        position = _run_user_signal(code, df)
        return ("ok", position)
    except Exception as e:
        return ("err", str(e))


def _run_in_process(code: str, df_bytes: bytes, queue: Any) -> None:
    """供 multiprocessing 调用的顶层函数（可 pickle），将结果放入 queue。"""
    try:
        status, payload = _worker_run_signal(code, df_bytes)
        queue.put((status, payload))
    except Exception as e:
        queue.put(("err", str(e)))


def run_custom_backtest(
    df: pd.DataFrame,
    code: str,
    initial_cash: float = 100_000.0,
    commission_rate: float = 0.0,
    timeout_seconds: float = 15.0,
) -> BacktestResult:
    """
    执行用户 Python 代码得到持仓序列，再跑回测。
    - code: 必须定义 signal(df)，返回与 df 同长度的 0/1（1=持多，0=空仓）。
    - 在子进程中执行并设超时；仅允许安全 builtins + pandas/numpy。
    """
    if len(code) > 50_000:
        raise ValueError("代码长度不能超过 50000 字符")
    if timeout_seconds <= 0 or timeout_seconds > 60:
        timeout_seconds = 15.0
    buf = io.BytesIO()
    df.to_pickle(buf)
    df_bytes = buf.getvalue()
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_run_in_process, args=(code, df_bytes, q))
    proc.start()
    proc.join(timeout=timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        raise TimeoutError(f"策略执行超时（限 {timeout_seconds} 秒）")
    if not q.empty():
        status, payload = q.get_nowait()
    else:
        raise RuntimeError("子进程未返回结果")
    if status == "err":
        raise ValueError(payload)
    position = payload
    return _run_from_position(df, position, initial_cash, commission_rate)
