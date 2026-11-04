/**
 * Backtest workspace UI — result tabs, governance, decision audit, multi-strategy.
 */
(function (global) {
  "use strict";

  const STATE = {
    resultMode: "single",
    rebalanceLogs: [],
    selectedAuditIndex: 0,
    symbolNameMap: {},
  };

  function escapeHtml(s) {
    if (s == null) return "";
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function formatPct(value, digits) {
    if (value == null || Number.isNaN(Number(value))) return "-";
    const n = Number(value) * 100;
    return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}%`;
  }

  function initResultTabs() {
    document.querySelectorAll(".result-tab").forEach((btn) => {
      btn.addEventListener("click", () => switchResultTab(btn.getAttribute("data-tab")));
    });
  }

  function switchResultTab(tabId) {
    if (!tabId) return;
    document.querySelectorAll(".result-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });
    document.querySelectorAll(".result-panel").forEach((panel) => {
      panel.classList.toggle("hidden", panel.getAttribute("data-panel") !== tabId);
    });
  }

  function setResultMode(mode) {
    STATE.resultMode = mode;
    const isPortfolio = mode === "portfolio";
    const isMulti = mode === "multi";
    const isSingle = mode === "single";

    document.querySelectorAll(".result-tab[data-mode]").forEach((tab) => {
      const modes = (tab.getAttribute("data-mode") || "").split(",");
      const visible = modes.some((m) => {
        if (m === "portfolio") return isPortfolio;
        if (m === "multi") return isMulti;
        if (m === "single") return isSingle;
        return false;
      });
      tab.classList.toggle("is-hidden", !visible);
    });

    const singleOnly = document.getElementById("singleOnlyCharts");
    if (singleOnly) singleOnly.classList.toggle("hidden", !isSingle);

    if (isMulti) switchResultTab("allocation");
    else if (isPortfolio) switchResultTab("overview");
    else switchResultTab("overview");
  }

  function renderGovernancePanel(data) {
    const el = document.getElementById("governancePanel");
    if (!el) return;
    const gov = data.governance_artifacts || {};
    const prov = data.run_provenance || gov.run_provenance || {};
    const regime = gov.regime_metrics || {};
    const stress = gov.stress_validation || {};
    const byRegime = regime.by_regime || {};

    const regimeRows = Object.entries(byRegime)
      .map(([label, m]) => {
        const ret = m.total_return != null ? formatPct(m.total_return, 2) : "-";
        const dd = m.max_drawdown != null ? formatPct(m.max_drawdown, 2) : "-";
        const wr = m.win_rate != null ? formatPct(m.win_rate, 1) : "-";
        return `<tr>
          <td>${escapeHtml(label)}</td>
          <td class="${Number(m.total_return) >= 0 ? "positive" : "negative"}">${ret}</td>
          <td>${dd}</td>
          <td>${wr}</td>
          <td>${m.turnover != null ? Number(m.turnover).toFixed(0) : "-"}</td>
          <td>${m.cost != null ? Number(m.cost).toFixed(2) : "-"}</td>
        </tr>`;
      })
      .join("");

    const stressRows = (stress.scenarios || [])
      .map((s, i) => {
        const diag = (stress.diagnostics || [])[i] || {};
        const failed = diag.failed;
        const ret = s.metrics?.total_return;
        return `<tr>
          <td>${escapeHtml(s.scenario)}</td>
          <td>${ret != null ? formatPct(ret, 2) : "-"}</td>
          <td>${failed ? '<span class="badge-fail">未通过</span>' : '<span class="badge-pass">通过</span>'}</td>
        </tr>`;
      })
      .join("");

    el.innerHTML = `
      <div class="gov-grid">
        <div class="gov-card">
          <h4>运行溯源</h4>
          <p class="sub">数据 / 代码 / 参数版本绑定</p>
          <ul class="kv-list">
            <li><span class="k">模式</span><span class="v">${escapeHtml(prov.mode || "backtest")}</span></li>
            <li><span class="k">数据版本</span><span class="v">${escapeHtml(prov.dataset_version || data.dataset_version || "-")}</span></li>
            <li><span class="k">代码版本</span><span class="v">${escapeHtml(prov.code_version || "-")}</span></li>
            <li><span class="k">参数 Hash</span><span class="v" style="font-size:0.72rem;">${escapeHtml(prov.params_hash || data.request_hash || "-")}</span></li>
            <li><span class="k">样本区间</span><span class="v">${escapeHtml(prov.sample_start || "")} ~ ${escapeHtml(prov.sample_end || "")}</span></li>
          </ul>
        </div>
        <div class="gov-card">
          <h4>压力测试</h4>
          <p class="sub">${stress.passed ? '<span class="badge-pass">整体通过</span>' : '<span class="badge-fail">存在未通过场景</span>'}</p>
          <table style="width:100%; font-size:0.8rem;">
            <thead><tr><th>场景</th><th>压力收益</th><th>状态</th></tr></thead>
            <tbody>${stressRows || "<tr><td colspan='3' style='color:var(--muted);'>暂无压力测试数据</td></tr>"}</tbody>
          </table>
        </div>
      </div>
      <div class="gov-card" style="margin-bottom:0;">
        <h4>Regime 指标切片</h4>
        <p class="sub">按人工配置市场状态窗口汇总（${escapeHtml(regime.market || "cn")}）</p>
        <div style="overflow-x:auto;">
          <table style="width:100%; font-size:0.82rem;">
            <thead>
              <tr>
                <th>Regime</th><th>收益</th><th>最大回撤</th><th>胜率</th><th>换手</th><th>成本</th>
              </tr>
            </thead>
            <tbody>${regimeRows || "<tr><td colspan='6' style='color:var(--muted);'>样本区间与 regime 窗口无交集</td></tr>"}</tbody>
          </table>
        </div>
      </div>`;
  }

  function renderDecisionAudit(rebalanceLogs, nameMap) {
    STATE.rebalanceLogs = rebalanceLogs || [];
    STATE.symbolNameMap = nameMap || {};
    STATE.selectedAuditIndex = 0;
    const listEl = document.getElementById("auditDayList");
    const detailEl = document.getElementById("auditDayDetail");
    if (!listEl || !detailEl) return;

    if (!STATE.rebalanceLogs.length) {
      listEl.innerHTML = "<li style='padding:12px;color:var(--muted);'>无调仓记录</li>";
      detailEl.innerHTML = "<p style='color:var(--muted);'>请先运行全市场组合回测。</p>";
      return;
    }

    listEl.innerHTML = STATE.rebalanceLogs
      .map((r, i) => {
        const rejected = r.rejected_orders || 0;
        return `<li class="audit-day-item${i === 0 ? " active" : ""}" data-index="${i}">
          <div><strong>${escapeHtml(r.trade_date)}</strong></div>
          <div class="meta">信号 ${escapeHtml(r.signal_date)} · 目标 ${r.target_count} · 拒单 ${rejected}</div>
        </li>`;
      })
      .join("");

    listEl.querySelectorAll(".audit-day-item").forEach((item) => {
      item.addEventListener("click", () => {
        const idx = parseInt(item.getAttribute("data-index"), 10);
        selectRebalanceDay(idx);
      });
    });

    selectRebalanceDay(0);
  }

  function selectRebalanceDay(index) {
    STATE.selectedAuditIndex = index;
    const r = STATE.rebalanceLogs[index];
    const detailEl = document.getElementById("auditDayDetail");
    const listEl = document.getElementById("auditDayList");
    if (!r || !detailEl) return;

    listEl?.querySelectorAll(".audit-day-item").forEach((el) => {
      el.classList.toggle("active", parseInt(el.getAttribute("data-index"), 10) === index);
    });

    const targets = (r.targets || [])
      .slice(0, 20)
      .map(
        (t) =>
          `<li>${formatSymbolLabel(t.symbol, STATE.symbolNameMap)} — 权重 ${(Number(t.weight) * 100).toFixed(2)}%，score ${Number(t.score || 0).toFixed(4)}，${escapeHtml(t.reason || "无备注")}</li>`
      )
      .join("");

    const rejected = (r.rejected_details || [])
      .map((x) => `<li>${formatSymbolLabel(x.symbol, STATE.symbolNameMap)} ${escapeHtml(x.side)}：${escapeHtml(x.reason)}</li>`)
      .join("");

    const cost = r.cost_snapshot || {};

    detailEl.innerHTML = `
      <div class="decision-tree">
        <div class="node">
          <div class="node-title">1. 数据与 Universe</div>
          <ul>
            <li>数据版本：<strong>${escapeHtml(r.dataset_version || "-")}</strong></li>
            <li>信号日：${escapeHtml(r.signal_date)} → 执行日：${escapeHtml(r.trade_date)}</li>
            <li>当日可交易池规模：${r.universe_size ?? "-"}</li>
          </ul>
        </div>
        <div class="node">
          <div class="node-title">2. 目标持仓（策略信号 + 排名）</div>
          <ul>${targets || "<li class='muted'>无目标</li>"}</ul>
        </div>
        <div class="node">
          <div class="node-title">3. 成本假设</div>
          <ul class="muted">
            <li>佣金(买/卖)：${cost.buy_commission_rate} / ${cost.sell_commission_rate}</li>
            <li>印花税：${cost.stamp_tax_rate}，滑点 ${cost.slippage_bps} bps</li>
          </ul>
        </div>
        <div class="node">
          <div class="node-title">4. 执行结果</div>
          <ul>
            <li>成交订单：${r.filled_orders ?? 0}，拒单：${r.rejected_orders ?? 0}</li>
            ${rejected ? `<li>拒单明细：</li></ul><ul>${rejected}</ul>` : "</ul>"}
        </div>
      </div>`;
  }

  function renderMultiStrategyResult(data) {
    const el = document.getElementById("allocationPanel");
    if (!el) return;
    const alloc = data.allocation || {};
    const weights = alloc.actual_weights || alloc.target_weights || {};
    const scores = alloc.inputs?.strategy_scores || {};
    const maxW = Math.max(...Object.values(weights).map(Number), 0.001);

    const bars = Object.entries(weights)
      .map(([sid, w]) => {
        const pct = Number(w) * 100;
        const width = (Number(w) / maxW) * 100;
        return `<div class="alloc-bar-row">
          <span class="alloc-bar-label">${escapeHtml(sid)}</span>
          <div class="alloc-bar-track"><div class="alloc-bar-fill" style="width:${width}%"></div></div>
          <span class="alloc-bar-pct">${pct.toFixed(1)}%</span>
        </div>`;
      })
      .join("");

    const sleeves = (data.sleeves || [])
      .map((s) => {
        const ret = s.metrics?.total_return;
        const cls = Number(ret) >= 0 ? "positive" : "negative";
        return `<div class="sleeve-card">
          <h5>${escapeHtml(s.sleeve_id)} <span style="font-weight:400;color:var(--muted);">(${escapeHtml(s.strategy)})</span></h5>
          <ul class="kv-list">
            <li><span class="k">总收益</span><span class="v ${cls}">${ret != null ? formatPct(ret, 2) : "-"}</span></li>
            <li><span class="k">子运行 Hash</span><span class="v" style="font-size:0.7rem;">${escapeHtml(s.request_hash || "-")}</span></li>
          </ul>
        </div>`;
      })
      .join("");

    el.innerHTML = `
      <p style="color:var(--muted); font-size:0.85rem; margin:0 0 12px 0;">
        各 sleeve 独立回测后，由 Allocator（${escapeHtml(data.allocator_method || "equal_weight")}）分配资金权重，禁止 naive 叠加。
      </p>
      <div class="gov-card">
        <h4>资本分配</h4>
        <p class="sub">约束：单策略最大 ${(alloc.constraints?.max_single_weight * 100 || 60).toFixed(0)}%，总敞口 ≤ ${(alloc.constraints?.max_gross_exposure * 100 || 100).toFixed(0)}%</p>
        <div class="alloc-bar-wrap">${bars || "<p class='muted'>无权重</p>"}</div>
        <ul class="kv-list">
          ${Object.entries(scores)
            .map(([k, v]) => `<li><span class="k">Score ${escapeHtml(k)}</span><span class="v">${Number(v).toFixed(4)}</span></li>`)
            .join("")}
        </ul>
      </div>
      <h4 style="font-size:0.95rem; margin:16px 0 8px 0;">Sleeve 回测摘要</h4>
      <div class="sleeve-cards">${sleeves}</div>`;
  }

  function formatSymbolLabel(symbol, nameMap) {
    const name = nameMap?.[symbol];
    if (!name || name === symbol) return escapeHtml(symbol);
    return `${escapeHtml(symbol)} · ${escapeHtml(name)}`;
  }

  function buildSymbolNameMap(data) {
    const attr = data?.governance_artifacts?.symbol_attribution || {};
    const map = { ...(attr.symbol_names || {}) };
    [...(attr.top_profit_contributors || []), ...(attr.top_drawdown_contributors || [])].forEach((row) => {
      if (row?.symbol) map[row.symbol] = row.name || row.symbol;
    });
    return map;
  }

  function renderCycleTimeline(cycles) {
    if (!cycles?.length) {
      return "<div style='color:var(--muted); font-size:0.8rem;'>暂无交易周期。</div>";
    }
    return cycles
      .map((cycle) => {
        const events = (cycle.events || [])
          .map((event) => {
            const sideLabel = event.side === "buy" ? "买入" : "卖出";
            return `<li>${escapeHtml(event.date)} ${sideLabel} ${Number(event.shares || 0).toLocaleString()} 股 @ ${Number(event.price || 0).toFixed(2)}，费用 ${Number(event.cost || 0).toFixed(2)}</li>`;
          })
          .join("");
        const statusLabel = cycle.status === "open" ? "持仓中" : "已平仓";
        const pnl = Number(cycle.realized_pnl || 0) + Number(cycle.unrealized_pnl || 0);
        return `<div style="margin:8px 0; padding:8px 10px; border:1px solid #30363d; border-radius:8px;">
          <div style="font-size:0.82rem; margin-bottom:4px;">
            周期 #${cycle.cycle_id} · ${statusLabel} · ${escapeHtml(cycle.entry_date || "-")} → ${escapeHtml(cycle.exit_date || "—")}
            · 盈亏 <span class="${pnl >= 0 ? "positive" : "negative"}">${pnl.toFixed(2)}</span>
          </div>
          <ul style="margin:0; padding-left:18px; font-size:0.8rem; color:var(--muted);">${events}</ul>
        </div>`;
      })
      .join("");
  }

  function renderAttributionTable(title, rows, { showDrawdown = false } = {}) {
    if (!rows?.length) {
      return `<div class="gov-card" style="margin-bottom:12px;">
        <h4 style="margin:0 0 8px 0;">${escapeHtml(title)}</h4>
        <p style="color:var(--muted); margin:0; font-size:0.85rem;">暂无数据（可能无成交或样本过短）。</p>
      </div>`;
    }
    const body = rows
      .map((row, index) => {
        const pnlCls = Number(row.total_pnl) >= 0 ? "positive" : "negative";
        const pnlSharePct = Number(row.pnl_share_pct || 0);
        const shareCls = pnlSharePct >= 0 ? "positive" : "negative";
        const ddDetail = showDrawdown
          ? `<td class="${shareCls}" title="该股盈亏占组合总盈亏的比例">${pnlSharePct.toFixed(1)}%</td>`
          : "";
        return `<tr class="attr-row" data-attr-index="${index}" data-attr-kind="${showDrawdown ? "drawdown" : "profit"}" style="cursor:pointer;">
          <td>${row.rank}</td>
          <td>${formatSymbolLabel(row.symbol, { [row.symbol]: row.name })}</td>
          <td class="${pnlCls}">${Number(row.total_pnl || 0).toFixed(2)}</td>
          <td class="${pnlCls}">${formatPct(row.total_return_pct, 2)}</td>
          ${ddDetail}
          <td>${row.trade_count ?? 0}</td>
          <td>${row.holding_days ?? 0}</td>
        </tr>
        <tr class="attr-detail hidden" data-detail-for="${showDrawdown ? "drawdown" : "profit"}-${index}">
          <td colspan="${showDrawdown ? 7 : 6}" style="background:#0d1117;">${renderCycleTimeline(row.cycles)}</td>
        </tr>`;
      })
      .join("");
    const ddRange = showDrawdown
      ? `<span style="font-size:0.75rem; color:var(--muted); font-weight:400; margin-left:6px;">(按实际盈亏排序)</span>`
      : "";
    const ddHeader = showDrawdown
      ? `<th title="该股盈亏占组合总盈亏的百分比">盈亏占比<sup style="color:var(--muted);cursor:help;" title="该股盈亏 / 组合总盈亏 × 100%">?</sup></th>`
      : "";
    return `<div class="gov-card" style="margin-bottom:12px;">
      <h4 style="margin:0 0 8px 0;">${escapeHtml(title)}${ddRange}</h4>
      <div style="overflow-x:auto;">
        <table style="width:100%; font-size:0.82rem;">
          <thead>
            <tr><th>#</th><th>标的</th><th>总盈亏</th><th>收益率</th>${ddHeader}<th>成交笔数</th><th>持仓日</th></tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      </div>
      <p style="color:var(--muted); font-size:0.75rem; margin:8px 0 0;">点击行展开完整买卖周期。</p>
    </div>`;
  }

  function renderSymbolAttribution(data) {
    const el = document.getElementById("symbolAttributionPanel");
    if (!el) return;
    const attr = data?.governance_artifacts?.symbol_attribution;
    if (!attr) {
      el.innerHTML = "";
      return;
    }
    const dd = attr.portfolio_drawdown || {};
    const totalPnl = attr.total_portfolio_pnl;
    const totalPnlStr = totalPnl != null ? `组合总盈亏 ${totalPnl >= 0 ? "+" : ""}${Number(totalPnl).toLocaleString(undefined, {maximumFractionDigits: 0})}` : "";
    el.innerHTML = `
      <div style="margin-bottom:8px; color:var(--muted); font-size:0.82rem;">
        组合最大回撤 ${formatPct(dd.max_drawdown, 2)}，
        窗口 ${escapeHtml(dd.drawdown_window_start || "-")} → ${escapeHtml(dd.drawdown_window_end || "-")}
        ${totalPnlStr ? `<span style="margin-left:8px;">· ${totalPnlStr}</span>` : ""}
      </div>
      ${renderAttributionTable("最大收益来源 Top 10", attr.top_profit_contributors || [])}
      ${renderAttributionTable("最大亏损来源 Top 10", attr.top_drawdown_contributors || [], { showDrawdown: true })}
    `;
    el.querySelectorAll(".attr-row").forEach((row) => {
      row.addEventListener("click", () => {
        const index = row.getAttribute("data-attr-index");
        const kind = row.getAttribute("data-attr-kind");
        const detail = el.querySelector(`tr.attr-detail[data-detail-for="${kind}-${index}"]`);
        if (detail) detail.classList.toggle("hidden");
      });
    });
  }

  function enhancePortfolioInsights(data) {
    const metrics = data.metrics || {};
    const wrEl = document.getElementById("insightWinRate");
    if (wrEl && metrics.win_rate != null) {
      wrEl.textContent = `${(Number(metrics.win_rate) * 100).toFixed(1)}%（组合日胜率）`;
      wrEl.style.color = Number(metrics.win_rate) >= 0.5 ? "var(--green)" : "var(--red)";
    }
    const ddEl = document.getElementById("insightDrawdown");
    if (ddEl && metrics.max_drawdown != null) {
      ddEl.textContent = `${formatPct(metrics.max_drawdown, 2)}（最大回撤）`;
    }
  }

  function afterPortfolioResult(data) {
    setResultMode("portfolio");
    renderGovernancePanel(data);
    renderSymbolAttribution(data);
    renderDecisionAudit(data.rebalance_logs || [], buildSymbolNameMap(data));
    enhancePortfolioInsights(data);
  }

  function afterMultiStrategyResult(data) {
    setResultMode("multi");
    renderMultiStrategyResult(data);
    if (data.sleeves?.[0]) {
      document.getElementById("strategyUsed").textContent =
        `多策略组合 · ${data.sleeves.length} sleeves · ${data.dataset_version || ""}`;
    }
  }

  global.BacktestWorkspace = {
    initResultTabs,
    switchResultTab,
    setResultMode,
    renderGovernancePanel,
    renderSymbolAttribution,
    renderDecisionAudit,
    renderMultiStrategyResult,
    afterPortfolioResult,
    afterMultiStrategyResult,
    enhancePortfolioInsights,
    buildSymbolNameMap,
    formatSymbolLabel,
    escapeHtml,
  };
})(window);
