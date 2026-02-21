#!/usr/bin/env python3
"""
大模型离线自检：在不上线 API 的前提下，在本机验证 Falcon-H1R-7B 能否加载并跑通一次推理。
通过后再启动 uvicorn，可避免线上报「大模型未就绪」。

用法（在项目根目录或 backend 目录执行）：
  python backend/scripts/test_llm_offline.py
  或
  cd backend && python scripts/test_llm_offline.py

若报 Network is unreachable / Errno 101（无法访问 Hugging Face），可改用本地已下载模型：
  1) 在能访问外网的机器或开 HTTP 代理下执行一次下载（仅需一次）：
     pip install huggingface_hub
     huggingface-cli download tiiuae/Falcon-H1R-7B --local-dir ./Falcon-H1R-7B
  2) 把 Falcon-H1R-7B 目录拷到本机，再设置环境变量后运行本脚本或启动 API：
     export FALCON_H1R_7B_PATH=/绝对路径/Falcon-H1R-7B
     python backend/scripts/test_llm_offline.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 代理相关环境变量（socks:// 会导致 transformers/huggingface_hub 报 Unknown scheme，加载时暂时取消）
_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

# 保证可导入 backend 下的 services（支持从项目根或 backend 目录执行）
_backend = Path(__file__).resolve().parent.parent
_root = _backend.parent
for _p in (_backend, _root):
    if _p and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main() -> int:
    print("=== 大模型离线自检（不连外网请求，仅本地加载与推理） ===\n")

    # Step 1: regex
    print("[1/4] 检查 regex ...")
    try:
        import regex
        print("      OK")
    except ImportError as e:
        print(f"      FAIL: {e}")
        print("      请执行: pip install regex")
        return 1

    # Step 2: transformers + 加载 pipeline（若环境有 socks 代理会报 Unknown scheme，此处暂时取消代理）
    from services.etf_llm import _get_model_path
    model_path = _get_model_path()
    if model_path == "tiiuae/Falcon-H1R-7B":
        print("[2/4] 加载 pipeline（未设置 FALCON_H1R_7B_PATH，将尝试从网络加载，需能访问 Hugging Face）...")
    else:
        print(f"[2/4] 从本地加载 pipeline：{model_path} ...")
    saved_proxy = {k: os.environ.pop(k, None) for k in _PROXY_KEYS}
    try:
        from services.etf_llm import _load_pipeline
        pipe, err = _load_pipeline()
        if err or pipe is None:
            print(f"      FAIL: {err or '加载失败'}")
            print("      无网络时请先离线下载模型并设置：export FALCON_H1R_7B_PATH=/path/to/Falcon-H1R-7B")
            return 1
        print(f"      OK (model={model_path})")
    except Exception as e:
        print(f"      FAIL: {e}")
        print("      建议: pip install -U 'transformers>=4.46.0' 'accelerate>=0.33.0' torch")
        return 1
    finally:
        for k, v in saved_proxy.items():
            if v is not None:
                os.environ[k] = v

    # Step 3: 用假摘要跑一次推理（不请求行情接口）
    print("[3/4] 跑一次推理（假摘要）...")
    fake_summary = {
        "symbol": "SPY",
        "start_date": "2019-01-01",
        "end_date": "2024-01-01",
        "total_return_pct": 80.5,
        "annual_return_pct": 12.3,
        "volatility_pct": 18.2,
        "sharpe_ratio": 0.95,
        "max_drawdown_pct": -20.1,
    }
    try:
        from services.etf_llm import run_llm_score
        result = run_llm_score("SPY", fake_summary)
        print(f"      OK -> score={result.get('score')}, reason={result.get('reason', '')[:80]}...")
    except Exception as e:
        print(f"      FAIL: {e}")
        return 1

    print("[4/4] 自检通过，可启动 API（如 uvicorn app.main:app），大模型评分类接口应可用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
