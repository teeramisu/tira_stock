# feat: 因子库 + jq-0020/0001/0003/0009 策略适配

- 新增 `backend/factors/primitives/` 统一 RSI、BBI、动量、择时 gate、ETF 排序等原语
- 落地组合策略：`jq_etf_momentum_volume_gate_top1`、`jq_wide_base_etf_bbi_rotation`、`jq_etf_value_averaging`、`jq_wpetf_momentum_top1`
- 单标的：`jq_etf_value_averaging_timing`
- 更新 `docs/JOINQUANT_ONE_CLICK_ADAPT.md` 与前端 `backtest-strategy-guide.js`
