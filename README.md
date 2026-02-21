# 美股股票回测系统

前后端分离的美股回测系统：**后端 FastAPI 只提供 REST API**，**前端单页**负责参数配置、发起回测、展示净值曲线与交易记录。

## 功能

- **市场与数据**：支持 **美股、A股、港股**。通过 yfinance 拉取 OHLCV，**支持最长 10 年区间**（6mo / 1y / 2y / 5y / 10y 或自定义起止日期）。代码格式：美股 AAPL；A股 600519（沪）/000001（深）；港股 0700/9988。
- **多策略回测**：
  - 双均线（SMA）金叉/死叉
  - 双均线（EMA）金叉/死叉
  - RSI 超卖超买（可设周期、超卖/超买阈值）
  - MACD 金叉死叉（可设快/慢/信号线周期）
  - 布林带（触及下轨买、上轨卖）
  - 买入持有（基准）
- **指标**：总收益、年化收益、夏普比率、最大回撤、波动率、交易次数、胜率。
- **ETF 评估**：多只 ETF/股票同一区间买入持有对比，总收益/年化/波动/夏普/最大回撤及归一化净值曲线（起点=100）。
- **前端**：单页表单（策略与参数选择）+ 净值曲线图（Chart.js）+ 交易明细表 + ETF 对比评估。

## 项目结构

```
stock_backtest/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI 入口与 /api 路由
│   ├── backtest/
│   │   ├── engine.py        # 回测引擎（可单测验证）
│   │   └── __init__.py
│   ├── services/
│   │   ├── data.py          # 拉取行情（yfinance）
│   │   └── __init__.py
│   └── tests/
│       └── test_engine.py    # 引擎单元测试
├── frontend/
│   └── index.html           # 单页前端
├── requirements.txt
├── run.sh
└── README.md
```

## 安装与运行

### 1. 安装依赖

```bash
cd stock_backtest
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 方式一
./run.sh

# 方式二
uvicorn main:app --app-dir backend/app --host 0.0.0.0 --port 8000
```

### 3. 使用

- 浏览器打开：**http://localhost:8000/** → 填写股票代码、日期/周期、均线周期后点击「运行回测」。
- API 文档：**http://localhost:8000/docs**

## API 说明

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/health | 健康检查 |
| GET | /api/strategies | 可用策略列表及参数名 |
| GET | /api/history | 查询历史 OHLCV，参数：symbol, start, end 或 period（含 10y） |
| POST | /api/backtest | 执行回测，Body：symbol, start/end/period(6mo~10y), strategy, initial_cash 及策略参数 |
| POST | /api/backtest/custom | **自定义 Python 策略**：Body 含 symbol, period, initial_cash, **code**。code 中须定义 `signal(df)`，返回与 df 同长的 0/1 序列（1=持多 0=空仓）。仅允许 pandas/numpy 及安全内置，执行限时 15 秒。 |
| POST | /api/etf/compare | **ETF 评估**：Body 含 symbols（2～20 个代码）、period、market。返回各标的买入持有指标及归一化净值曲线。 |

## 回测逻辑（可验证）

- 使用**前一日**的均线信号，**当日收盘**成交，避免未来数据。
- 仅做多：金叉全仓买入，死叉全仓卖出。
- 可选手续费率 `commission_rate`（按成交金额比例）。
- 单元测试见 `backend/tests/test_engine.py`，可直接运行验证：
  ```bash
  python backend/tests/test_engine.py
  ```

## 注意事项

- 数据来源为 yfinance（Yahoo Finance），仅供学习与回测研究。
- 回测结果不代表实盘表现，实盘需考虑滑点、流动性、交易成本等。
