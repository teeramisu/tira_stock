# 工程化评估：完备性与改进建议

## 总体结论

当前项目是**可用的 MVP / 演示级**应用：功能完整、回测逻辑可验证、数据库与 API 已搭好。距离**工程级（可上线、可运维、可扩展）**还缺：标准化部署、观测与安全、测试与文档等一整套实践。

---

## 一、已具备的工程化要素

| 方面 | 现状 |
|------|------|
| **架构** | 前后端分离、REST API、Pydantic 请求校验 |
| **回测** | 引擎纯函数、可单测验证，多策略 + 自定义 Python 策略 |
| **数据** | 异步 PostgreSQL、SQLAlchemy 2、Alembic 迁移、连接池、未配置时降级 |
| **安全（策略）** | 自定义代码子进程 + 超时 + 受限 builtins，避免任意执行 |
| **文档** | README 使用说明、DATABASE.md 表结构与迁移说明、FastAPI /docs |

---

## 二、不完备之处与改进建议

### 1. 版本与仓库

- **无 `.gitignore`**：`.venv/`、`__pycache__/`、`*.pyc`、`.env`、`alembic.ini` 中的敏感占位等易被提交。
- **建议**：增加 `.gitignore`，并避免把 `.env`、本地 `alembic.ini` 覆盖提交。

### 2. 配置管理

- **无 `.env.example`**：`DATABASE_URL` 等仅文档提及，新人/部署不知要配哪些变量。
- **建议**：新增 `.env.example`，列出 `DATABASE_URL`、可选 `LOG_LEVEL` 等，不含真实密码。

### 3. 日志与可观测性

- **无统一 logging**：异常仅通过 HTTP 返回，无日志文件或聚合，排障困难。
- **建议**：在 `main.py` 及 `data.py`、`custom_strategy.py` 等处使用 `logging`，按请求 id 或 symbol 打关键步骤与异常；生产可输出 JSON 便于采集。

### 4. 部署与运行环境

- **无 Docker / 编排**：依赖、Python 版本、启动方式未固化，多环境不一致。
- **建议**：提供 `Dockerfile`（多阶段构建）和可选 `docker-compose.yml`（app + postgres），文档中说明 `docker build` / `docker compose up`。

### 5. 测试

- **仅有引擎单测**：`backend/tests/test_engine.py` 覆盖核心回测逻辑，无 API 层、无 DB 层。
- **无 pytest 配置**：无 `pytest.ini` 或 `pyproject.toml` 的 `[tool.pytest.ini_options]`，无 `conftest.py`（如 client、db fixture）。
- **建议**：  
  - 用 pytest 跑现有单测，并加 `tests/test_api_*.py`（TestClient 调用 `/api/backtest`、`/api/health` 等）。  
  - 需要时加 `tests/test_db_*.py`（内存 SQLite 或 testcontainers PostgreSQL），测试保存/拉取回测记录。  
  - 在 `requirements-dev.txt` 中加 pytest、httpx、pytest-asyncio 等。

### 6. 安全

- **CORS**：`allow_origins=["*"]`，生产应限定前端域名。
- **无认证**：`/api/backtest/save`、`/api/history/records` 未鉴权，任何人可写/查。
- **无限流**：未防刷，自定义策略接口易被滥用。
- **错误信息**：部分 `detail=str(e)` 可能泄露内部路径或依赖信息。
- **建议**：  
  - 生产 CORS 改为具体 origin；  
  - 为“保存/历史”等接口加 JWT 或 API Key，并与 `user_id` 关联；  
  - 对 `/api/backtest` 与 `/api/backtest/custom` 做按 IP 或按 user 的限流；  
  - 对外返回统一错误码与简短文案，详细错误仅写日志。

### 7. 健康检查

- **`/api/health` 未检 DB**：仅返回固定文案，无法判断“应用可运行但数据库挂了”。
- **建议**：若配置了 `DATABASE_URL`，在 health 中执行一次简单查询（如 `SELECT 1`），失败时返回 503 或 `healthy: false`。

### 8. 依赖

- **仅 `requirements.txt`**：无版本上限，无开发/测试依赖分离。
- **建议**：核心依赖可加版本上限（如 `sqlalchemy>=2.0,<3`）；另设 `requirements-dev.txt`（pytest、httpx、pytest-asyncio、black、ruff 等），CI 与本地开发使用。

### 9. 前端

- **单文件 794 行**：功能集中在一个 HTML，后续难以分工与复用。
- **历史记录仅 localStorage**：未对接 `POST /api/backtest/save` 与 `GET /api/history/records`，多设备/清缓存即丢失。
- **无构建与类型**：无打包、无 TypeScript/类型检查、无前端单测。
- **建议**：  
  - 若保持单页，可先拆成独立 js/css 文件按功能模块划分；  
  - 增加“保存到云端”按钮并调用保存 API，历史栏支持从“云端记录”加载；  
  - 若计划长期迭代，可考虑 Vite/React 等并引入类型与测试。

### 10. 文档与 README

- **README 未写数据库与保存/历史 API**：新人不知要配库、跑迁移、有哪些新接口。
- **建议**：在 README 中增加一节“数据库（可选）”：`DATABASE_URL`、`alembic upgrade head`、`POST /api/backtest/save`、`GET /api/history/records` 的简要说明，并指向 `docs/DATABASE.md`。

### 11. CI/CD

- **无自动化流水线**：无提交时跑测试、无 lint、无迁移检查。
- **建议**：用 GitHub Actions（或同类）在 push/PR 时：安装依赖、运行 pytest、可选 ruff/black 检查；若有 Docker，可加镜像构建。

### 12. 数据库与运维

- **数据保留策略**：`backtest_runs` 会持续增长，文档虽提“按 created_at 归档”，未实现。
- **建议**：后续可加定时任务或管理接口，按时间删除或归档旧回测记录；必要时对 `result_summary` 做大小/字段约束，避免单行过大。

---

## 三、按优先级可做的下一步

1. **必做（达到“可上线”底线）**  
   - 增加 `.gitignore`、`.env.example`。  
   - 健康检查包含 DB 连通性（若配置了 DB）。  
   - 生产 CORS 收紧；保存/历史接口加简单认证或 API Key。  
   - README 补充数据库与保存/历史 API 说明。

2. **建议做（提升可维护性）**  
   - 统一 logging；对外错误信息收敛，详细错误只写日志。  
   - 用 pytest 跑全量测试，并增加 API 集成测试。  
   - Dockerfile + docker-compose，文档中写清部署步骤。

3. **可选（随规模再补）**  
   - 限流、前端对接保存/历史 API、CI 流水线、前端拆分或重构。

---

## 四、小结

| 维度       | 现状     | 工程级目标     |
|------------|----------|----------------|
| 功能       | 已满足   | 已满足         |
| 测试       | 仅引擎   | 引擎 + API + 可选 DB |
| 配置/环境  | 文档说明 | .env.example + Docker |
| 安全       | 宽松     | CORS + 认证 + 限流 + 错误收敛 |
| 可观测性   | 无       | 日志 + 健康检查含 DB |
| 文档       | 部分     | README 含 DB/API + 部署说明 |
| CI/CD      | 无       | 提交跑测试 + 可选构建 |

按上述清单逐步补齐后，可视为**工程级**可上线、可运维的回测系统。
