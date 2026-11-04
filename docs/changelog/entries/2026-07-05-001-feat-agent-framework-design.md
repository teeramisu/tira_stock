# [feat] 多 Agent 投资分析框架 — 设计文档 v0.3.0

- **日期**：2026-07-05
- **版本**：v0.3.0-dev
- **类型**：feat (design)

## 背景

现有系统 LLM 使用分散（`etf_llm.py`、`stock_recommendation.py`），无统一 Agent 框架、无记忆系统、无多信号融合。目标是构建自动化股票分析提醒机器人。

## 调研结论

### Memory 架构

| 方案 | 结论 |
|------|------|
| Letta/MemGPT | 3 层记忆 + Git 版本化，但太重 |
| CrewAI Memory | 复合评分（语义+近因+重要性），但需 LanceDB |
| LangGraph Store | Checkpointer + Store 分离，需手搭层级 |
| **自建 DuckDB** | **零新依赖，与现有行情仓库统一，3 张表搞定** |

### Agent 框架

| 方案 | 结论 |
|------|------|
| CrewAI | Role+Goal+Backstory 内置编排，但需外部 LLM API |
| OpenAI Agents SDK | Session+Handoff 强绑 OpenAI |
| **自建** | **~800 行，完全控制，复用 Falcon-H1R-7B 本地模型，现有 services 就是工具** |

## 架构设计

### Agent 结构

```
Orchestrator (Supervisor + DAG)
  ├── SentimentAgent    (新闻情绪 → 0-100 分)
  ├── IndustryAgent     (行业轮动 → 0-100 分)
  ├── TechniqueAgent    (量价因子 → 0-100 分)
  ├── RiskGate          (风控过滤)
  └── StrategyAgent     (信号融合 → buy/sell/权重)
```

### 三层记忆

| 层级 | 存储 | 内容 | 召回 | 保留 |
|------|------|------|------|------|
| Hot | Agent 实例属性 | 持仓、市场状态、风险参数 | 始终在上下文 | Session |
| Warm | DuckDB `agent_signals` | 信号+推理+置信度+结果 | 复合评分（近因 0.6 + 语义 0.3 + 重要性 0.1） | 90 天 |
| Cold | DuckDB `agent_decisions` | 调优历史、策略表现 | 语义搜索 | 永久 |

### 信号流

```
News/Market/Factors → [Sentiment|Industry|Technique]Agent
                          ↓ Signal(score+confidence+reasoning)
                     Memory (入库)
                          ↓
                    StrategyAgent (融合)
                          ↓ RiskGate (风控)
                          ↓ Decision (buy/sell/weight/stop-loss)
                     Memory (入库)
                          ↓
                    API → 前端 Dashboard
```

### DuckDB 新增表

```sql
agent_signals     -- 信号记录 (agent_name, signal_type, score, confidence, reasoning, ticker, timestamp)
agent_decisions   -- 决策记录 (run_id, agent_name, decision, position, stop_loss, reasoning, performance_after)
market_snapshots  -- 市场快照 (trade_date, regime, vix_level, snapshot JSON)
```

### 工具系统

| Tool | 复用 |
|------|------|
| `news_tool` | `services/news.py` |
| `industry_tool` | `industry_heat.py` + `industry_board.py` |
| `market_data_tool` | DuckDB + `services/data.py` |
| `technical_tool` | `factors/primitives/*` |
| `backtest_tool` | `engine.py` / `portfolio/engine.py` |

### 自建模块清单

```
backend/
├── agents/         # Agent 框架 (~250 loc)
│   ├── base.py     # BaseAgent 抽象
│   ├── registry.py # Agent 注册表
│   ├── orchestrator.py # DAG + Supervisor 编排
│   ├── sentiment.py    # 情绪面 Agent
│   ├── industry.py     # 行业面 Agent
│   ├── technique.py    # 技术面 Agent
│   ├── strategy.py     # 策略融合 Agent
│   └── risk.py         # 风控 Gate
├── memory/         # 记忆系统 (~200 loc)
│   ├── models.py       # Pydantic 模型
│   ├── store.py        # DuckDB CRUD
│   └── session.py      # In-memory short-term
├── tools/           # 工具 (~150 loc)
│   ├── base.py         # BaseTool 抽象
│   ├── news.py         # 新闻工具
│   ├── industry.py     # 行业数据工具
│   ├── market_data.py  # 行情数据工具
│   └── technical.py    # 技术指标工具
└── signals/         # 信号系统 (~100 loc)
    ├── models.py        # Signal + SignalHub
    └── combiner.py      # 置信度加权融合
```

### 自动化提醒

```
Cron (每日 9:00 / 15:30)
  → Agent Pipeline (6 agents)
    → SignalHub (融合)
      → 信号 > 阈值？→ 前端 Dashboard 展示
      → 信号入库（供回测/复盘）
```

### API 端点（新增）

| Method | Path | 说明 |
|--------|------|------|
| POST | `/api/agents/run` | 运行全部 Agent 分析，返回信号列表 |
| POST | `/api/agents/run/{agent_name}` | 运行单个 Agent |
| GET | `/api/memory/signals` | 查询历史信号（agent_name, since, limit） |
| GET | `/api/memory/decisions` | 查询历史决策 |
| GET | `/api/agents/status` | 各 Agent 最新运行状态 |

### 前端集成

在现有 `index.html` 的 "Data Readiness" 区域下方新增 **Agent Dashboard** 卡片：
- 各 Agent 最近信号分数（情绪分 / 行业分 / 技术分 / 综合分）
- 信号推理理由
- 触发"刷新分析"按钮

## 实施计划

| Phase | 内容 | 工作量 |
|-------|------|--------|
| 1 | Memory 系统（DuckDB 表 + Schemas + Store） | 0.5 天 |
| 2 | Agent 框架（BaseAgent + Registry + Orchestrator） | 0.5 天 |
| 3 | 工具系统（包装现有 services） | 0.5 天 |
| 4 | 首批 Agent（Sentiment + Strategy + RiskGate） | 1 天 |
| 5 | API + 前端 Agent Dashboard | 0.5 天 |

## 参考

- CrewAI Memory 复合评分机制：`semantic_weight * similarity + recency_weight * decay + importance_weight * importance`
- LangGraph 层级记忆：Checkpointer (短期) + Store (长期) 分离
- FinGPT RLSP：用实际股价走势作为 reward signal 驱动模型优化
- Generative Agents 反思模式：定期从 memory stream 中提取高层 insight
