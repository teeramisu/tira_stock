# 🦞 Tira 炒股小帮手

> 你的专属AI股票助手，基于OpenClaw + 美股回测系统构建

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.11+-green)
![OpenClaw](https://img.shields.io/badge/OpenClaw-2026.3.7-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 项目简介

**Tira 炒股小帮手** 是一只智能股票管理AI助手，通过Telegram与用户交互，帮助你：

- 📊 查询全球股票历史行情（美股、A股、港股）
- 📈 执行策略回测分析
- 📉 多ETF对比评估
- 📰 获取实时市场新闻
- 🔔 主动推送重要信息

---

## 🦞 这只龙虾能做什么？

### 1️⃣ 股票数据查询

支持获取任意股票的历史OHLCV数据，最长可达10年。

| 市场 | 代码示例 | 说明 |
|------|----------|------|
| 美股 | `AAPL`, `MSFT`, `GOOGL` | 苹果、微软、谷歌 |
| A股 | `600519`, `000001` | 茅台、平安银行 |
| 港股 | `0700`, `9988` | 腾讯、阿里巴巴 |

### 2️⃣ 智能回测分析

支持多种技术指标策略回测：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| SMA Cross | 双均线金叉死叉 | 趋势跟踪 |
| EMA Cross | 指数均线交叉 | 敏感交易 |
| RSI | 超卖超买区间 | 反转交易 |
| MACD | 金叉死叉+背离 | 中长线 |
| Bollinger | 布林带突破 | 区间震荡 |
| Buy & Hold | 买入持有 | 基准对比 |

### 3️⃣ ETF对比分析

支持2-20只ETF同时对比，买入持有策略：

- 总收益率
- 年化收益率
- 波动率
- 夏普比率
- 最大回撤
- 归一化净值曲线

### 4️⃣ 低估ETF推荐

基于P/E、P/B、股息率综合打分，智能推荐相对低估的ETF。

### 5️⃣ 市场新闻

实时获取各市场新闻：

- 美股：WSJ、CNBC等
- A股：上交所指数新闻
- 港股：东方财富、新浪财经
- 期货：东方财富

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Node.js 18+ (用于OpenClaw)
- 网络访问（API数据源：Yahoo Finance）

### 1. 克隆项目

```bash
cd ~
git clone https://github.com/your-repo/stock_backtest.git
cd stock_backtest
```

### 2. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# Windows: .venv\Scripts\activate

# 安装Python依赖
pip install -r requirements.txt
```

### 3. 启动股票API服务

```bash
# 方式一：使用脚本
./run.sh

# 方式二：手动启动
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### 4. 配置OpenClaw（可选）

```bash
# 检查OpenClaw状态
openclaw status

# 启动网关
openclaw gateway
```

---

## 📱 Telegram 使用指南

### 启动龙虾

在Telegram中搜索你的机器人（`@tiras_cat_bot`），发送以下关键词即可唤醒：

- 🦞
- 股票
- stock
- 龙虾

### 常用命令

```
# 查询股票历史
帮我查一下苹果的股票走势
AAPL 最近一年的数据

# 执行回测
用SMA策略回测AAPL
RSI策略分析腾讯

# ETF对比
对比SPY和QQQ
帮我看看哪些ETF被低估

# 获取新闻
今天美股有什么新闻
港股最新消息
```

---

## 🔧 API 接口

### 基础信息

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/strategies` | 可用策略列表 |

### 股票数据

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/history` | 查询历史OHLCV |
| POST | `/api/backtest` | 执行回测 |
| POST | `/api/backtest/custom` | 自定义Python策略 |

### ETF分析

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/etf/compare` | ETF对比评估 |
| GET | `/api/etf/undervalued` | 低估ETF推荐 |
| POST | `/api/etf/llm-score` | ETF大模型评分 |

### 新闻资讯

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/news/us` | 美股新闻 |
| GET | `/api/news/cn` | A股新闻 |
| GET | `/api/news/hk` | 港股新闻 |
| GET | `/api/news/futures` | 期货新闻 |

### 完整请求示例

```bash
# 查询历史数据
curl "http://localhost:8000/api/history?symbol=AAPL&period=1y&market=us"

# 执行回测
curl -X POST "http://localhost:8000/api/backtest" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "market": "us",
    "period": "1y",
    "strategy": "sma_crossover",
    "fast_period": 10,
    "slow_period": 30,
    "initial_cash": 100000
  }'

# ETF对比
curl -X POST "http://localhost:8000/api/etf/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["SPY", "QQQ", "VOO"],
    "period": "5y",
    "market": "us"
  }'
```

---

## 🛠️ 项目结构

```
stock_backtest/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI入口
│   ├── backtest/
│   │   ├── engine.py        # 回测引擎
│   │   └── custom_strategy.py  # 自定义策略
│   ├── services/
│   │   ├── data.py          # 行情数据获取
│   │   ├── etf.py           # ETF评估
│   │   ├── news.py          # 新闻获取
│   │   └── etf_valuation.py # 低估ETF
│   ├── database/
│   │   ├── models.py        # 数据库模型
│   │   └── session.py       # 数据库会话
│   └── tests/
│       └── test_engine.py   # 单元测试
├── frontend/
│   └── index.html          # Web界面
├── scripts/
│   ├── stock_helper.sh      # 命令行工具
│   └── lobster_report.sh    # 定时推送脚本
├── requirements.txt         # Python依赖
├── run.sh                   # 启动脚本
└── README.md                # 项目说明
```

---

## ⚙️ 高级配置

### 修改默认参数

编辑 `backend/app/main.py` 中的请求模型：

```python
class BacktestRequest(BaseModel):
    initial_cash: float = Field(100_000.0, ...)  # 默认资金
    commission_rate: float = Field(0.001, ...)     # 手续费率
```

### 添加自定义策略

使用 `/api/backtest/custom` 接口提交Python代码：

```python
def signal(df):
    # df 包含: Open, High, Low, Close, Volume
    ma20 = df['Close'].rolling(20).mean()
    ma60 = df['Close'].rolling(60).mean()
    return (ma20 > ma60).astype(int)  # 返回0/1序列
```

### 配置定时推送

编辑 `~/.openclaw/cron/jobs.json`：

```json
{
  "jobs": [
    {
      "id": "lobster-daily",
      "schedule": "0 9 * * *",
      "command": "./scripts/lobster_report.sh daily"
    }
  ]
}
```

---

## 📊 回测示例

### SMA策略回测AAPL（1年）

```
策略: SMA Cross (10/30)
标的: AAPL (苹果)
周期: 2025-03-18 ~ 2026-03-18

📈 收益分析
─────────────
初始资金:     $100,000.00
最终价值:     $120,154.19
总收益:       +20.15%
年化收益:     +20.15%

📉 风险指标
─────────────
波动率:       23.45%
夏普比率:     0.86
最大回撤:     -15.32%
回撤持续:     45天

📊 交易统计
─────────────
交易次数:     6
盈利次数:     4
胜率:         66.67%
```

---

## ⚠️ 免责声明

- 本系统仅供学习与研究使用，不构成投资建议
- 回测结果不代表实盘表现
- 实际交易需考虑滑点、流动性、交易成本等因素
- 数据来源：Yahoo Finance，可能存在延迟或不准确

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m 'Add xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建Pull Request

---

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 了解详情

---

## 🙏 致谢

- [OpenClaw](https://openclaw.ai) - AI Agent框架
- [yfinance](https://pypi.org/project/yfinance/) - 股票数据源
- [FastAPI](https://fastapi.tiangolo.com/) - Web框架
- [pandas](https://pandas.pydata.org/) - 数据分析

---

<div align="center">

**🦞 Tira 炒股小帮手 - 你的智能股票管家 🦞**

Made with ❤️ by Tira

</div>