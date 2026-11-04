/**
 * Strategy algorithm guide — rendered below the run-backtest controls.
 */
(function (global) {
  "use strict";

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  const EXEC_SINGLE = [
    "信号在 T 日收盘后计算，使用 T 日及之前数据（无前视）。",
    "T+1 日按收盘价调仓：position 从 0→1 全仓买入，1→0 全仓卖出。",
    "单标的、整数股数按可用现金尽可能买满；手续费按成交金额比例扣除。",
  ];

  const EXEC_PORTFOLIO_CN = [
    "信号日 T 计算目标权重，T+1 日开盘价撮合（无 lookahead）。",
    "A 股：买入须为 100 股整数倍；卖出须为 100 股整数倍，清仓时可一次性卖出全部持仓（含零股）。",
    "美股：按 1 股为单位；默认过滤 ST、停牌、涨跌停拒单等规则。",
    "等权目标权重 → Broker 计算买卖股数 → 成本含佣金/印花税/滑点。",
  ];

  const GUIDES = {
    sma_crossover: {
      title: "双均线 SMA 金叉 / 死叉",
      source: "backend/backtest/engine.py → `_sma_crossover_signal`",
      intro:
        "计算快、慢简单移动平均线（SMA）。当昨日快线 ≥ 慢线时持多（1），否则空仓（0）。经典趋势跟踪：金叉做多、死叉平仓。",
      params: ["fast_period", "slow_period"],
      code: `fast_ma = close.rolling(fast_period).mean()
slow_ma = close.rolling(slow_period).mean()
# 昨日快线 >= 昨日慢线 → 持多
position = (fast_ma.shift(1) >= slow_ma.shift(1)).astype(int)`,
    },
    ema_crossover: {
      title: "双均线 EMA 金叉 / 死叉",
      source: "backend/backtest/engine.py → `_ema_crossover_signal`",
      intro: "与 SMA 相同逻辑，但使用指数加权均线（EMA），对近期价格更敏感。",
      params: ["fast_period", "slow_period"],
      code: `fast_ma = close.ewm(span=fast_period).mean()
slow_ma = close.ewm(span=slow_period).mean()
position = (fast_ma.shift(1) >= slow_ma.shift(1)).astype(int)`,
    },
    rsi: {
      title: "RSI 超卖 / 超买",
      source: "backend/backtest/engine.py → `_rsi_signal`",
      intro:
        "用 Wilder 风格 RSI（滚动均值增益/损失）。RSI 跌入超卖区（默认 <30）开多；升入超买区（>70）平仓；中间区间保持原仓位。",
      params: ["period_rsi", "oversold", "overbought"],
      code: `rsi = 100 - 100 / (1 + avg_gain / avg_loss)
if prev_rsi < oversold:  position = 1
elif prev_rsi > overbought: position = 0
else: keep previous position`,
    },
    macd: {
      title: "MACD 金叉 / 死叉",
      source: "backend/backtest/engine.py → `_macd_signal`",
      intro: "MACD 线 = EMA(快) − EMA(慢)；信号线 = MACD 的 EMA。昨日 MACD > 信号线时持多。",
      params: ["macd_fast", "macd_slow", "macd_signal"],
      code: `macd = ema(close, fast) - ema(close, slow)
signal = ema(macd, signal_period)
position = (macd.shift(1) > signal.shift(1)).astype(int)`,
    },
    bollinger: {
      title: "布林带均值回归",
      source: "backend/backtest/engine.py → `_bollinger_signal`",
      intro: "中轨 = SMA(N)，上下轨 = 中轨 ± k·σ。价格触及下轨买入，触及上轨卖出，中间保持仓位。",
      params: ["bollinger_period", "num_std"],
      code: `mid = close.rolling(period).mean()
upper, lower = mid + k*std, mid - k*std
if prev_close <= lower: buy
elif prev_close >= upper: sell`,
    },
    buy_hold: {
      title: "买入持有",
      source: "backend/backtest/engine.py",
      intro: "首日全仓买入后不再交易，作为被动基准策略。",
      params: [],
      code: `position = [0, 1, 1, 1, ...]  # 第二根 K 线起持多`,
    },
    jq_bbi_momentum_timing: {
      title: "JoinQuant BBI 动量择时（jq-0001 单标的）",
      source: "backend/backtest/imported_strategies.py → `jq_bbi_momentum_signal`",
      intro:
        "BBI = 多周期 SMA 均值（默认 3/6/12/24）。收盘价强于 BBI 且日收益率不低于阈值时持多。完整多 ETF 轮动请用组合版 `jq_wide_base_etf_bbi_rotation`。",
      params: ["ma1", "ma2", "ma3", "ma4", "min_daily_return"],
      code: `bbi = mean(SMA(close, p) for p in [3,6,12,24])
position = (prev_close > prev_bbi) & (prev_ret >= min_daily_return)`,
    },
    jq_rsrs_timing: {
      title: "JoinQuant RSRS 择时",
      source: "backend/backtest/imported_strategies.py → `jq_rsrs_timing_signal`",
      intro:
        "用 High~Low 线性回归斜率衡量支撑/阻力强度，标准化为 z-score 并乘以 R²。分数 > 阈值且价格在均线上方开多；分数 < −阈值或跌破均线清仓。",
      params: ["rsrs_n", "rsrs_m", "rsrs_threshold", "rsrs_ma_period"],
      code: `slope = linreg(high, low, n)
score = zscore(slope, m) * R2
if score > threshold and close >= MA: long
elif score < -threshold or close < MA: flat`,
    },
    jq_rsi_gap_reversal: {
      title: "JoinQuant RSI 衍生 ETF 择时",
      source: "backend/backtest/imported_strategies.py → `jq_rsi_gap_reversal_signal`",
      intro:
        "对 log(Close/High) 求 RSI，结合趋势均线、K 线实体形态（十字星过滤）及止盈止损逻辑生成 0/1 持仓序列。",
      params: ["gap_rsi_period", "gap_rsi_buy", "gap_rsi_sell", "stop_loss", "take_profit"],
      code: `rsi = RSI(log(close / high))
+ MA trend filter + body-shape filter
+ trailing stop_loss / take_profit`,
    },
    momentum_top_k: {
      title: "动量 Top K（全市场组合）",
      source: "backend/portfolio/strategies.py → `MomentumTopKStrategy`",
      intro:
        "每个调仓日，在可交易 Universe 内按 lookback 日收益率排名，选取 Top K 标的，等权配置（权重 = 1/K）。动量因子 = last_close/first_close − 1。",
      params: ["portfolio_lookback_days", "portfolio_max_positions"],
      code: `score[s] = close[s, T] / close[s, T-L] - 1
top_k = scores.nlargest(K)
weight = 1 / len(top_k)  # 等权`,
      exec: EXEC_PORTFOLIO_CN,
    },
    etf_rotation_top1: {
      title: "ETF / 指数轮动 Top1",
      source: "backend/portfolio/strategies.py → `EtfRotationTop1Strategy`",
      intro: "在 Universe（通常填 ETF 代码池）中选 lookback 收益最高的一只，100% 权重集中持有。",
      params: ["portfolio_lookback_days"],
      code: `score[s] = close[s, T] / close[s, T-L] - 1
hold argmax(score) with weight = 1.0`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_value_ma_rsi_timing: {
      title: "JoinQuant 价值投资大盘择时（jq-0002 择时层）",
      source: "backend/backtest/imported_strategies.py → `jq_value_ma_rsi_timing_signal`",
      intro:
        "源自「54.价值投资策略-大盘择时」的 RiskControl：MA 乖离率状态机 + 多周期 RSI。建议在 510300 等宽基 ETF 上回测；不含原策略基本面选股。",
      params: ["slow_period"],
      code: `ma_rate = close / MA(close, period)
# 状态 NORMAL / WARNING 切换
if WARNING: RSI(15), RSI(90) 过滤
else: RSI(60) 过滤 → 持多或空仓`,
    },
    jq_value_timing_top_k: {
      title: "JoinQuant 价值投资择时组合（jq-0002 无财报 proxy）",
      source: "backend/portfolio/strategies.py → `JqValueTimingTopKStrategy`",
      intro:
        "无财报版 jq-0002：每月按成交额 proxy 排序选股，510300 上 MA+RSI 大盘 gate 关闭时不持仓；买入过滤含 5 日跌幅与均线结构。周频调仓，默认 5 只等权。",
      params: ["portfolio_lookback_days", "portfolio_max_positions"],
      code: `# 每月候选 = avg(amount) 排序 + 60日收益带
# gate(510300) 为真才持仓
# 买入前: ret_5d >= -10%, 非 judge_More_average
top_k equal weight`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_etf_polyfit_momentum_top1: {
      title: "JoinQuant ETF polyfit 动量 Top1（jq-0019）",
      source: "backend/portfolio/jq_strategies.py → `EtfPolyfitMomentumTop1Strategy`",
      intro: "jq-0019：对 log(close) 做线性回归，score=年化斜率×R²，在自定义 ETF 池中持最强 1 只。",
      params: ["portfolio_lookback_days", "portfolio_symbols"],
      code: `score = annualized(log(close) ~ t) * R²
hold argmax(score), weight=1`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_etf_momentum_rsrs_top1: {
      title: "JoinQuant ETF 动量+RSRS Top1（jq-0010/0008）",
      source: "backend/portfolio/jq_strategies.py → `EtfMomentumRsrsTop1Strategy`",
      intro: "polyfit 动量选 1 只 ETF；510300 RSRS zscore×R² + MA 端点比较 gate 为假则空仓。",
      params: ["portfolio_lookback_days", "portfolio_symbols"],
      code: `if rsrs_ma_gate(510300): hold argmax(polyfit_score)
else: empty`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_small_cap_top_k: {
      title: "JoinQuant 小市值 TopK（jq-0006 无财报）",
      source: "backend/portfolio/jq_strategies.py → `JqSmallCapTopKStrategy`",
      intro: "成交额升序 proxy 小市值，可选 MA5 站上才买；不含 valuation.market_cap 与国九条 filter。",
      params: ["portfolio_lookback_days", "portfolio_max_positions"],
      code: `candidates = sort asc avg(amount)
filter close >= MA5
top_k equal weight`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_etf_ma_cross_timing: {
      title: "JoinQuant 简单 ETF MA 交叉（jq-0026）",
      source: "backend/backtest/imported_strategies.py → `jq_etf_ma_cross_timing_signal`",
      intro: "MA60 向上 + 价低于 MA60 + MA5 向上买入；MA20 下穿 MA10 卖出。",
      params: ["slow_period", "fast_period"],
      code: `buy: MA60↑ & close<MA60 & MA5↑
sell: cross_down(MA20, MA10)`,
    },
    jq_industry_etf_bbi_rotation: {
      title: "JoinQuant 行业ETF BBI轮动+择时（jq-0007）",
      source: "backend/portfolio/jq_strategies.py → `JqIndustryEtfBbiRotationStrategy`",
      intro:
        "大盘广度 + 510300 EMA5 空头则持 511880；否则在 ETF 池中选 BBI/价最低（最偏多）行业 ETF 满仓。周频调仓。",
      params: ["portfolio_symbols"],
      code: `if ema5_bearish(510300) or !market_breadth: hold 511880
else: hold argmin(BBI/price) ETF where ratio<1`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_wide_base_etf_bbi_rotation: {
      title: "JoinQuant 宽基ETF BBI轮动（jq-0001 完整版）",
      source: "backend/portfolio/jq_strategies.py → `JqWideBaseEtfBbiRotationStrategy`",
      intro:
        "jq-0001 完整池：多宽基 ETF 日频 BBI(3/6/12/24) 选 BBI/价最低；大盘广度（最强指数日涨幅>0）为真才持仓。策略库一键优先此组合版。",
      params: ["portfolio_symbols"],
      code: `if breadth_best_return(proxies) > 0:
  hold argmin(BBI/price) among wide-base ETF pool
else: flat`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_etf_momentum_volume_gate_top1: {
      title: "JoinQuant ETF动量+成交量gate（jq-0020）",
      source: "backend/portfolio/jq_strategies.py → `JqEtfMomentumVolumeGateTop1Strategy`",
      intro:
        "13 日涨幅排序 + 均线溢价过滤；510300 成交量情绪或 510500 价格情绪恶化时持 511880；ETF 广度弱时同样避险。",
      params: ["portfolio_lookback_days", "portfolio_symbols"],
      code: `if volume_emotion(510300)==-1 or price_emotion(510500)<0: bond
elif breadth_weak(etf_pool): bond
else: hold top1 passing jq0020 filters`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_etf_value_averaging: {
      title: "JoinQuant ETF价值平均定投（jq-0003 proxy）",
      source: "backend/portfolio/jq_strategies.py → `JqEtfValueAveragingStrategy`",
      intro:
        "月度调仓 proxy：指数回撤入场（rate1<0.318）后逐步加仓权重；收益>40%且回撤>9%或收益>100%止盈清仓。非真实申购金额 DCA。",
      params: ["portfolio_symbols"],
      code: `monthly:
  if !active and pullback_allows: weight=0.7
  elif active: weight *= month_growth_rate
  take_profit rules → flat`,
      exec: EXEC_PORTFOLIO_CN,
    },
    jq_etf_value_averaging_timing: {
      title: "JoinQuant ETF价值平均择时（jq-0003 单标的）",
      source: "backend/backtest/imported_strategies.py → `jq_etf_value_averaging_timing_signal`",
      intro: "单标的 proxy：回撤入场 + 移动止盈；不含月度申购与 IRR。推荐 159905 等宽基/红利 ETF。",
      params: ["entry_pullback_rate"],
      code: `if pullback_rate < 0.318: enter long
if ret>40% and dd>9% or ret>100%: exit`,
    },
    jq_wpetf_momentum_top1: {
      title: "JoinQuant WPETF动量 Top1（jq-0009 ETF腿）",
      source: "backend/portfolio/jq_strategies.py → `WpetfMomentumTop1Strategy`",
      intro:
        "jq-0009 仅 ETF 子策略：log(close) 多项式动量×R² + 曲线/RSI 修正，持得分最高 1 只；不含大白马/小市值腿与资金分配。",
      params: ["portfolio_lookback_days", "portfolio_symbols"],
      code: `score = wpetf_momentum_score(close, m_days=25)
hold argmax(score) where score > min_score`,
      exec: EXEC_PORTFOLIO_CN,
    },
    portfolio_multi: {
      title: "多策略组合（Multi-Sleeve + Allocator）",
      source: "backend/services/multi_strategy_portfolio.py",
      intro:
        "多个独立 sleeve 各自跑完整组合回测（如快/慢动量），得到各 sleeve 收益曲线与指标；Allocator 按 equal_weight 等方法分配总资金权重，禁止 naive 直接叠加权重。",
      params: ["multi_lookback_a", "multi_lookback_b", "multi_max_single_weight"],
      code: `for sleeve in sleeves:
    metrics[sleeve] = backtest(momentum_top_k, lookback=sleeve.L)
weights = allocator.equal_weight(metrics, max_single_weight)`,
      exec: EXEC_PORTFOLIO_CN,
    },
  };

  function readParam(id, fallback) {
    const el = document.getElementById(id);
    if (!el) return fallback;
    const v = el.value;
    return v === "" ? fallback : v;
  }

  function paramSummary(keys) {
    if (!keys?.length) return "";
    const labels = {
      fast_period: "快线",
      slow_period: "慢线",
      period_rsi: "RSI周期",
      oversold: "超卖",
      overbought: "超买",
      macd_fast: "MACD快",
      macd_slow: "MACD慢",
      macd_signal: "信号线",
      bollinger_period: "布林周期",
      num_std: "标准差倍数",
      portfolio_lookback_days: "回看天数",
      portfolio_max_positions: "最大持仓",
      multi_lookback_a: "Sleeve A回看",
      multi_lookback_b: "Sleeve B回看",
      multi_max_single_weight: "单策略最大权重",
    };
    return keys
      .map((k) => {
        const idMap = {
          fast_period: "fast_period",
          slow_period: "slow_period",
          period_rsi: "period_rsi",
          oversold: "oversold",
          overbought: "overbought",
          macd_fast: "macd_fast",
          macd_slow: "macd_slow",
          macd_signal: "macd_signal",
          bollinger_period: "bollinger_period",
          num_std: "num_std",
          portfolio_lookback_days: "portfolio_lookback_days",
          portfolio_max_positions: "portfolio_max_positions",
          multi_lookback_a: "multi_lookback_a",
          multi_lookback_b: "multi_lookback_b",
          multi_max_single_weight: "multi_max_single_weight",
        };
        const elId = idMap[k] || k;
        return `${labels[k] || k}=${readParam(elId, "-")}`;
      })
      .join(" · ");
  }

  function resolveGuideKey() {
    const mode = document.getElementById("backtestMode")?.value || "single";
    if (mode === "portfolio_multi") return "portfolio_multi";
    if (mode === "portfolio") {
      return document.getElementById("portfolio_strategy")?.value || "momentum_top_k";
    }
    return document.getElementById("strategy")?.value || "sma_crossover";
  }

  function render() {
    const body = document.getElementById("strategyGuideBody");
    if (!body) return;
    const key = resolveGuideKey();
    const guide = GUIDES[key] || GUIDES.sma_crossover;
    const mode = document.getElementById("backtestMode")?.value || "single";
    const market = document.getElementById("market")?.value || "us";
    const execRules =
      guide.exec || (mode === "single" ? EXEC_SINGLE : EXEC_PORTFOLIO_CN);
    const modeLabel =
      mode === "portfolio"
        ? "全市场组合"
        : mode === "portfolio_multi"
          ? "多策略组合"
          : "单标的";

    // 马卡龙色工具
    const _mc = [
      {bg:'#f8c8d8',fg:'#8b3a50'},{bg:'#fdd8a9',fg:'#7a5520'},{bg:'#d4e8b4',fg:'#3a6b20'},
      {bg:'#b8d8e8',fg:'#205a6b'},{bg:'#d8c8e8',fg:'#5a3a7a'},{bg:'#f8e8a0',fg:'#6b5a10'},
      {bg:'#c8e8d0',fg:'#206b3a'},{bg:'#f0b8c8',fg:'#7a2040'},{bg:'#b8c8f0',fg:'#20407a'},
    ];
    function _mcTag(text) {
      let h = 0;
      for (let i = 0; i < text.length; i++) h = ((h << 5) - h + text.charCodeAt(i)) | 0;
      const c = _mc[Math.abs(h) % _mc.length];
      return `<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:${c.bg};color:${c.fg};font-size:0.75rem;margin:2px 4px 2px 0;font-weight:600;">${text}</span>`;
    }

    body.innerHTML = `
      <div class="strategy-guide-meta">
        ${_mcTag(modeLabel)}
        ${_mcTag(MARKET_LABELS[market] || market)}
        <span class="strategy-guide-source">${escapeHtml(guide.source)}</span>
      </div>
      <h3 class="strategy-guide-title">${escapeHtml(guide.title)}</h3>
      <p class="strategy-guide-intro">${escapeHtml(guide.intro)}</p>
      ${
        guide.params?.length
          ? `<p class="strategy-guide-params"><strong>当前参数：</strong>${escapeHtml(paramSummary(guide.params))}</p>`
          : ""
      }
      <h4 class="strategy-guide-subhead">执行规则</h4>
      <ul class="strategy-guide-list">${execRules.map((r) => `<li>${escapeHtml(r)}</li>`).join("")}</ul>
      <h4 class="strategy-guide-subhead">核心逻辑（伪代码）</h4>
      <pre class="strategy-guide-code"><code>${escapeHtml(guide.code)}</code></pre>
    `;
  }

  const MARKET_LABELS = { cn: "A 股", us: "美股", hk: "港股" };

  global.BacktestStrategyGuide = { render, GUIDES };
})(window);
