"""
美股回测系统 API：前后端分离，仅提供 REST 接口。
"""
from __future__ import annotations

import os
import logging

import sys

# 加载 .env 文件（DEEPSEEK_API_KEY 等环境变量）
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(_env_path):
        load_dotenv(_env_path)
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# 保证可导入 backend 包（routers 等）
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.routers import backtest, marketdata, etf, strategy_agents, rl_agent, discovery, macro

app = FastAPI(
    title="美股回测系统 API",
    description="提供历史行情与双均线策略回测接口",
    version="1.0.0",
)

logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端单页：/ 返回 index.html，避免覆盖 /api
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
INDEX_PATH = os.path.join(FRONTEND_DIR, "index.html")
WORKSPACE_JS_PATH = os.path.join(FRONTEND_DIR, "js", "backtest-workspace.js")
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _workspace_asset_ver() -> str:
    try:
        return str(int(os.path.getmtime(WORKSPACE_JS_PATH)))
    except OSError:
        return "0"


@app.middleware("http")
async def disable_frontend_cache(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/assets/"):
        for key, value in NO_CACHE_HEADERS.items():
            response.headers[key] = value
    return response


@app.get("/")
def serve_index():
    if os.path.isfile(INDEX_PATH):
        with open(INDEX_PATH, encoding="utf-8") as f:
            html = f.read()
        asset_ver = _workspace_asset_ver()
        html = html.replace("__WORKSPACE_ASSET_VER__", asset_ver)
        return HTMLResponse(html, headers=NO_CACHE_HEADERS)
    return {"message": "美股回测 API 运行中。请将 frontend/index.html 放到前端服务器或访问 /api/health"}


if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="frontend_assets")


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "stock-backtest"}


# ---------- 注册路由 ----------
app.include_router(backtest.router)
app.include_router(marketdata.router)
app.include_router(etf.router)
app.include_router(strategy_agents.router)
app.include_router(rl_agent.router)
app.include_router(discovery.router)
app.include_router(macro.router)
