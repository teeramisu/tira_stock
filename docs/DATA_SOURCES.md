# 股票历史数据源说明

本项目中行情数据优先使用 **Yahoo Finance (yfinance)**，在限流或失败且为**美股**时，会依次尝试以下备用源。所有备用源仅支持美股代码（如 AAPL、SPY）。

## 已接入的备用源（按尝试顺序）

| 数据源 | 环境变量 | 说明 | 申请/文档 |
|--------|----------|------|----------|
| **Alpha Vantage** | `ALPHAVANTAGE_API_KEY` | 免费约 5 次/分钟、500 次/天；compact 约 100 个交易日 | [申请 Key](https://www.alphavantage.co/support/#api-key) |
| **Stooq** | 无 | 无需 Key，需安装 `pandas-datareader` | 已用 pandas-datareader 拉取 |
| **Financial Modeling Prep (FMP)** | `FMP_API_KEY` | 免费档有每日请求限制，历史数据较全 | [Dashboard](https://site.financialmodelingprep.com/developer/docs/dashboard) |

配置示例（在 `.env` 或当前 shell 中）：

```bash
# 可选，多配多一层保障
export ALPHAVANTAGE_API_KEY=你的key
export FMP_API_KEY=你的key
```

Stooq 只需安装依赖，无需配置：

```bash
pip install pandas-datareader
```

## 其他常见免费/稳定数据源（未接入）

以下为常见选项，可按需自行对接或替换：

- **Polygon.io**：免费档有限，[文档](https://polygon.io/docs/stocks/get_v2_aggs_ticker__stocksticker__range__multiplier___timespan___from___to)
- **Twelvedata**：免费档有限，[文档](https://twelvedata.com/docs)
- **IEX Cloud**：免费档有限，[文档](https://iexcloud.io/docs/)
- **Tiingo**：曾提供免费历史数据，[文档](https://www.tiingo.com/documentation/end-of-day)
- **Tushare**：A 股为主，需注册，[tushare.pro](https://tushare.pro)
- **AkShare**：A 股/港股等，无需 Key，`pip install akshare`

若需支持 A 股/港股历史，可考虑在 `data_sources.py` 中为 `market=cn` / `market=hk` 增加 Tushare 或 AkShare 的拉取函数，并在 `data.py` 的 fallback 分支中按市场调用。
