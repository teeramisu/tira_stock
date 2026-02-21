# 数据库设计说明（万级用户）

## 1. 选型与容量

| 项目 | 建议 |
|------|------|
| **数据库** | PostgreSQL 14+（支持 JSONB、并发好、易扩展） |
| **驱动** | asyncpg（异步，连接池） |
| **ORM/迁移** | SQLAlchemy 2.0（async）+ Alembic |
| **容量估算** | 1 万用户 × 每人约 50 条回测 ≈ 50 万行 `backtest_runs`，单库足够；百万级再考虑分表/只读副本 |

## 2. 表结构

### 2.1 用户表 `users`（为后续登录预留）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGSERIAL PRIMARY KEY | 主键 |
| email | VARCHAR(255) UNIQUE NOT NULL | 登录邮箱 |
| password_hash | VARCHAR(255) NOT NULL | 密码哈希 |
| name | VARCHAR(128) | 昵称 |
| created_at | TIMESTAMPTZ DEFAULT now() | 注册时间 |
| updated_at | TIMESTAMPTZ DEFAULT now() | 更新时间 |

- 索引：`email`（唯一）、`created_at`（可选，按注册时间查）

### 2.2 回测记录表 `backtest_runs`（核心）

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGSERIAL PRIMARY KEY | 主键 |
| user_id | BIGINT REFERENCES users(id) ON DELETE CASCADE | 用户 ID，可空表示未登录/匿名 |
| anonymous_id | VARCHAR(64) | 匿名设备 ID（前端生成 UUID），未登录时关联 |
| symbol | VARCHAR(32) NOT NULL | 股票代码 |
| market | VARCHAR(8) NOT NULL | 市场 us/cn/hk |
| strategy | VARCHAR(64) NOT NULL | 策略标识 |
| params | JSONB | 请求参数（period/start/end/快慢线等） |
| result_summary | JSONB NOT NULL | 结果摘要：total_return, annual_return, n_trades, win_rate, final_value 等，不存完整净值曲线 |
| created_at | TIMESTAMPTZ DEFAULT now() | 创建时间 |

- **不存**：完整 `equity_curve`、逐笔 `trades`，避免单行过大；需要时可由前端再请求一次回测或后续单独存“快照”。
- 索引：
  - `(user_id, created_at DESC)`：用户维度查最近记录、分页
  - `(anonymous_id, created_at DESC)`：匿名用户查自己的记录
  - `created_at`：按时间清理冷数据或分区

### 2.3 可选：自定义策略保存表 `saved_strategies`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGSERIAL PRIMARY KEY | |
| user_id | BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE | |
| name | VARCHAR(128) NOT NULL | 策略名称 |
| code | TEXT NOT NULL | Python 代码 |
| created_at | TIMESTAMPTZ DEFAULT now() | |

- 索引：`user_id`。

## 3. 连接与池

- 使用 **SQLAlchemy 2.0 async** + **asyncpg**，单库连接池大小建议：`pool_size=20, max_overflow=10`（按实际 QPS 调整）。
- 万级用户、每人请求频率不高时，单 PostgreSQL 实例 + 连接池即可；后续可加 PgBouncer 或只读副本。

## 4. 环境变量

- `DATABASE_URL`：例如 `postgresql+asyncpg://user:pass@host:5432/dbname`
- 未配置时：接口可降级为“仅内存/仅前端 localStorage”，不写库。

## 5. 迁移

- 在项目根目录执行（需先设置 `DATABASE_URL`）：
  - `alembic upgrade head`：应用迁移，建表/更新表结构。
  - `alembic revision --autogenerate -m "描述"`：根据模型变更生成新迁移。
- 初始迁移已提供：`alembic/versions/001_initial_users_and_backtest_runs.py`，建表 `users`、`backtest_runs` 及索引。

## 6. 安全与清理

- 密码仅存 `password_hash`（如 bcrypt）。
- 定期按 `created_at` 归档或删除过旧 `backtest_runs`（例如只保留 1 年），控制表体积。
