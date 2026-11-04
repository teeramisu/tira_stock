#!/usr/bin/env python3
"""运行一轮 Discovery Loop（P0/P1）：真实数据 + purged WF + DSR/PBO + 账本。

示例:
    python scripts/run_discovery_cycle.py
    python scripts/run_discovery_cycle.py --symbols 600519,300750 --start 2018-01-01
    python scripts/run_discovery_cycle.py --market us --symbols AAPL,MSFT --start 2018-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from discovery import (  # noqa: E402
    DEFAULT_PARAM_GRIDS,
    SIGNAL_FACTORIES,
    ExperimentLedger,
    Panel,
    PortfolioGenome,
    causality_test,
    cost_stress_test,
    delay_stress_test,
    evaluate_param_grid,
    evaluate_portfolio_grid,
    expand_grid,
    load_panel,
    make_walk_forward_splits,
    panel_quality,
    portfolio_promotion_gate,
    promotion_gate,
    run_portfolio_backtest,
)
from discovery.evaluator import PurgedSplit  # noqa: E402
from services.data import fetch_ohlcv  # noqa: E402


def _consensus_params(folds: list[dict]) -> dict:
    """取各 fold 在 train 上选出的参数的众数，作为最终推荐参数。"""
    if not folds:
        return {}
    keys = [json.dumps(f["best_params"], sort_keys=True, ensure_ascii=False) for f in folds]
    best = Counter(keys).most_common(1)[0][0]
    return json.loads(best)


def _build_portfolio_genomes() -> list[PortfolioGenome]:
    """组合 Genome 网格：因子 × 回看 × 持仓数 × 权重方式 × 调仓频率。"""
    specs = [
        ("momentum", 60),
        ("momentum", 120),
        ("multi", 60),
        ("reversal", 20),
        ("low_vol", 20),
    ]
    genomes: list[PortfolioGenome] = []
    for factor, lookback in specs:
        for top_k in (10, 20, 30):
            for weighting in ("equal", "inv_vol"):
                for rebalance in (5, 20):
                    max_weight = max(0.08, min(0.25, 2.0 / top_k))
                    genomes.append(
                        PortfolioGenome(
                            name=f"{factor}{lookback}_k{top_k}_{weighting}_r{rebalance}",
                            factor=factor,
                            lookback=lookback,
                            top_k=top_k,
                            weighting=weighting,
                            rebalance_days=rebalance,
                            max_weight=max_weight,
                        )
                    )
    return genomes


def _run_portfolio(args, ledger: ExperimentLedger) -> int:
    """组合级正式评估：选股 → 权重 → 调仓 → 成本 → 组合净值。"""
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or None
    print(f"=== 组合级 Discovery: {args.market} 股票池={'自定义' if symbols else '流动性前 ' + str(args.universe_limit)} ===")
    try:
        panel = load_panel(
            market=args.market,
            symbols=symbols,
            start=args.start,
            end=args.end or None,
            limit=args.universe_limit,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"加载行情面板失败: {exc}")
        return 1

    quality = panel_quality(panel)
    if not quality["passed"]:
        print("⚠️ 数据质量警告：检测到全市场异常单日收益，可能是复权基准断层：")
        for item in quality["bad_dates"][:5]:
            print(f"   {item['date']}  横截面均值 {item['mean_return']*100:+.1f}%")
        print("   建议使用 --end 2026-07-10，或先修复复权数据。")

    eval_start = panel.close.index.searchsorted(pd.Timestamp(args.start))
    n_eval = len(panel.close) - eval_start
    base_splits = make_walk_forward_splits(
        n_eval,
        train_size=args.train,
        test_size=args.test,
        step_size=args.step,
        purge=args.purge,
        embargo=args.embargo,
    )
    splits = [
        PurgedSplit(fold=s.fold, train=s.train + eval_start, test=s.test + eval_start)
        for s in base_splits
    ]
    print(f"面板: {panel.close.shape[1]} 只 × {len(panel.close)} 交易日 "
          f"({panel.close.index.min().date()} ~ {panel.close.index.max().date()})")
    print(f"评估区间: {panel.close.index[eval_start].date()} 起，{len(splits)} 个 purged fold")
    if len(splits) < 2:
        print("fold 数不足 2，请缩短 --train/--test 或扩大数据区间")
        return 1

    genomes = _build_portfolio_genomes()
    print(f"候选 Genome: {len(genomes)} 个，开始组合回测/选参…")
    result = evaluate_portfolio_grid(panel, genomes, splits, cost_bps=args.cost_bps)

    if "error" in result:
        print(f"评估失败: {result['error']}")
        return 1

    agg = result["aggregate"]
    bench = result["benchmark"]
    pbo = (result.get("pbo") or {}).get("pbo")
    gate = result.get("gate", portfolio_promotion_gate(result))
    # 各 fold 选出的 Genome 众数，作为本轮代表候选
    names = [f["genome"]["name"] for f in result["folds"]]
    consensus_name = Counter(names).most_common(1)[0][0] if names else genomes[0].name
    consensus = next(g for g in genomes if g.name == consensus_name)
    data_as_of = str(panel.close.index.max().date())
    dataset_version = f"{args.market}-{data_as_of}"

    print("\n" + "=" * 110)
    print(f"组合级 OOS: Sharpe {agg['sharpe']:.2f} | CAGR {agg['cagr']*100:.1f}% | "
          f"MaxDD {agg['max_drawdown']*100:.1f}% | Calmar {agg['calmar']:.2f}")
    print(f"基准等权:   CAGR {bench['cagr']*100:.1f}% | Sharpe {bench['sharpe']:.2f} | "
          f"超额 CAGR {result['excess_cagr']*100:+.1f}%")
    print(f"统计闸门:   DSR {result.get('dsr') or 0:.3f} | PBO {pbo if pbo is not None else '-'} | "
          f"正fold {result['fold_positive_ratio']*100:.0f}% | 年换手 {result['annual_turnover']:.1f}x")
    print(f"代表 Genome: {consensus_name} | {consensus.to_dict()}")
    print(f"晋升结论:   {'PASS' if gate['passed'] else 'REJECT(' + ','.join(gate['failed']) + ')'}")
    print("-" * 110)
    print(f"{'fold':>4}  {'train Sharpe':>12}  {'test Sharpe':>11}  {'test CAGR':>9}  {'excess CAGR':>11}  genome")
    for f in result["folds"]:
        print(f"{f['fold']:>4}  {f['train_score']:>12.2f}  {f['test']['sharpe']:>11.2f}  "
              f"{f['test']['cagr']*100:>8.1f}%  {f['test_excess_cagr']*100:>10.1f}%  {f['genome']['name']}")
    print("=" * 110)

    genome_payload = {
        **consensus.to_dict(),
        "market": args.market,
        "universe_size": int(panel.close.shape[1]),
        "evaluation": "portfolio_walk_forward_ensemble",
    }
    cid = ledger.record_candidate(
        genome_payload,
        stage="validated" if gate["passed"] else "rejected",
        promotion_status="candidate" if gate["passed"] else "rejected",
    )
    ledger.record_experiment(
        genome_hash=cid,
        candidate_id=cid,
        strategy=f"portfolio:{consensus.factor}",
        params=consensus.to_dict(),
        dataset_version=dataset_version,
        split_id=f"portfolio-wf-{args.train}-{args.test}-p{args.purge}",
        metrics={
            "evaluation": "portfolio_walk_forward_ensemble",
            "aggregate": agg,
            "benchmark": bench,
            "excess_cagr": result["excess_cagr"],
            "excess_total_return": result["excess_total_return"],
            "annual_turnover": result["annual_turnover"],
            "fold_positive_ratio": result["fold_positive_ratio"],
            "fold_sharpe_mean": result["fold_sharpe_mean"],
            "folds": result["folds"],
            "gate": gate,
            "universe_size": int(panel.close.shape[1]),
            "data_as_of": data_as_of,
        },
        dsr=result.get("dsr"),
        pbo=pbo,
        status="completed" if gate["passed"] else "rejected",
        failure_reason="" if gate["passed"] else ",".join(gate["failed"]),
    )
    if not gate["passed"]:
        ledger.record_lesson(
            hypothesis=f"portfolio {consensus_name} universe={panel.close.shape[1]}",
            failure_mode="; ".join(gate["failed"]),
            evidence={"dsr": result.get("dsr"), "pbo": pbo, "aggregate": agg,
                      "excess_cagr": result["excess_cagr"]},
        )
    print(f"账本: {ledger.path} | candidate={cid}")
    return 0 if gate["passed"] else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Discovery Loop P0/P1 cycle")
    parser.add_argument("--mode", default="portfolio", choices=["portfolio", "signal"],
                        help="portfolio=组合级正式评估；signal=单票信号诊断")
    parser.add_argument("--symbols", default="", help="组合模式为空则按流动性取前 N；信号模式为空则用默认 3 只")
    parser.add_argument("--market", default="cn", choices=["cn", "us", "hk"])
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument("--train", type=int, default=252, help="训练窗口交易日")
    parser.add_argument("--test", type=int, default=63, help="测试窗口交易日")
    parser.add_argument("--step", type=int, default=63, help="滚动步长交易日")
    parser.add_argument("--purge", type=int, default=10, help="train/test 间隔交易日")
    parser.add_argument("--embargo", type=int, default=5, help="fold 间禁入交易日")
    parser.add_argument("--cost-bps", type=float, default=10.0, help="单边换手成本(bp)")
    parser.add_argument("--ledger", default=str(ROOT / "data" / "discovery" / "ledger.db"))
    parser.add_argument("--min-dsr", type=float, default=0.95)
    parser.add_argument("--max-pbo", type=float, default=0.2)
    parser.add_argument("--universe-limit", type=int, default=500, help="组合模式股票池大小（按流动性取前 N）")
    parser.add_argument("--min-sharpe", type=float, default=0.8)
    args = parser.parse_args()

    ledger = ExperimentLedger(args.ledger)
    if args.mode == "portfolio":
        return _run_portfolio(args, ledger)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = ["600519", "300750", "002594"]
    all_rows = []

    for symbol in symbols:
        print(f"\n=== {args.market}:{symbol} 加载行情 ===")
        try:
            df = fetch_ohlcv(
                symbol,
                start=args.start,
                end=args.end or None,
                market=args.market,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  跳过 {symbol}: {exc}")
            continue
        prices = df["Close"].astype(float).dropna()
        if len(prices) < args.train + args.test + 20:
            print(f"  跳过 {symbol}: 样本不足 ({len(prices)})")
            continue
        data_as_of = str(prices.index.max())[:10]
        dataset_version = f"{args.market}-{data_as_of}"
        splits = make_walk_forward_splits(
            len(prices),
            train_size=args.train,
            test_size=args.test,
            step_size=args.step,
            purge=args.purge,
            embargo=args.embargo,
        )
        print(f"  行情 {prices.index.min().date()} ~ {data_as_of}，{len(prices)} 根，{len(splits)} 个 fold")
        if len(splits) < 2:
            print("  跳过：fold 数不足 2")
            continue

        for name, factory in SIGNAL_FACTORIES.items():
            grid = expand_grid(DEFAULT_PARAM_GRIDS.get(name, {}))
            try:
                result = evaluate_param_grid(
                    prices, factory, grid, splits, cost_bps=args.cost_bps
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<10} 评估失败: {exc}")
                continue
            if "error" in result:
                print(f"  {name:<10} {result['error']}")
                continue

            params = _consensus_params(result["folds"])
            causal = causality_test(prices, factory, params, n_probes=5)
            delay = delay_stress_test(prices, factory, params, splits, cost_bps=args.cost_bps)
            cost = cost_stress_test(
                prices, factory, params, splits, cost_bps=args.cost_bps, multipliers=(1, 2, 3)
            )
            gate = promotion_gate(
                result,
                min_sharpe=args.min_sharpe,
                min_dsr=args.min_dsr,
                max_pbo=args.max_pbo,
            )

            agg = result["aggregate"]
            pbo_value = (result.get("pbo") or {}).get("pbo")
            genome = {
                "market": args.market,
                "symbol": symbol,
                "strategy": name,
                "params": params,
            }
            cid = ledger.record_candidate(
                genome,
                stage="validated" if gate["passed"] else "rejected",
                promotion_status="candidate" if gate["passed"] else "rejected",
            )
            metrics = {
                "symbol": symbol,
                "aggregate": agg,
                "fold_positive_ratio": result["fold_positive_ratio"],
                "fold_sharpe_mean": result["fold_sharpe_mean"],
                "fold_sharpe_std": result["fold_sharpe_std"],
                "causality": causal,
                "delay": delay,
                "cost_stress": cost,
                "gate": gate,
                "data_as_of": data_as_of,
            }
            ledger.record_experiment(
                genome_hash=cid,
                candidate_id=cid,
                strategy=name,
                params=params,
                dataset_version=dataset_version,
                split_id=f"wf-{args.train}-{args.test}-p{args.purge}",
                metrics=metrics,
                dsr=result.get("dsr"),
                pbo=pbo_value,
                status="completed" if gate["passed"] else "rejected",
                failure_reason="" if gate["passed"] else ",".join(gate["failed"]),
            )
            if not gate["passed"]:
                ledger.record_lesson(
                    hypothesis=f"{symbol}/{name} {params}",
                    failure_mode="; ".join(gate["failed"]),
                    evidence={"dsr": result.get("dsr"), "pbo": result.get("pbo"), "aggregate": agg},
                )

            all_rows.append({
                "symbol": symbol,
                "strategy": name,
                "params": params,
                "sharpe": agg["sharpe"],
                "cagr": agg["cagr"],
                "maxdd": agg["max_drawdown"],
                "dsr": result.get("dsr"),
                "pbo": pbo_value,
                "pos_folds": result["fold_positive_ratio"],
                "causal": causal.get("passed"),
                "delay": delay.get("retention"),
                "cost_ok": cost.get("passed"),
                "gate": gate["passed"],
                "failed": ",".join(gate["failed"]),
            })

    if not all_rows:
        print("\n没有产生有效实验。")
        return 1

    all_rows.sort(key=lambda r: (r["gate"], r["dsr"] or 0), reverse=True)
    print("\n" + "=" * 118)
    print(
        f"{'标的':<8}{'策略':<11}{'Sharpe':>8}{'CAGR':>8}{'MaxDD':>8}"
        f"{'DSR':>8}{'PBO':>7}{'正fold':>8}{'因果':>6}{'延迟':>7}{'成本':>6}  结果"
    )
    print("-" * 118)
    for r in all_rows:
        print(
            f"{r['symbol']:<8}{r['strategy']:<11}{r['sharpe']:>8.2f}{r['cagr']*100:>7.1f}%"
            f"{r['maxdd']*100:>7.1f}%{(r['dsr'] or 0):>8.3f}{(r['pbo'] or 0):>7.3f}"
            f"{r['pos_folds']*100:>7.0f}%{('✓' if r['causal'] else '✗'):>6}"
            f"{(r['delay'] or 0):>7.2f}{('✓' if r['cost_ok'] else '✗'):>6}  "
            f"{'PASS' if r['gate'] else 'REJECT(' + r['failed'] + ')'}"
        )
    print("=" * 118)
    print(f"账本: {ledger.path}")
    print(f"汇总: {ledger.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
