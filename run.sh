#!/bin/bash
# 启动美股回测系统（后端 + 前端单页）
cd "$(dirname "$0")"
if [ -d ".venv" ]; then
  . .venv/bin/activate
fi
uvicorn main:app --app-dir backend/app --host 0.0.0.0 --port 8000
