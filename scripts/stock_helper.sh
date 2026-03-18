#!/bin/bash
# Stock API Helper for Lobster Agent
# Usage: ./stock_helper.sh <command> [args...]

API_BASE="http://localhost:8000"

case "$1" in
  history)
    symbol="${2:-AAPL}"
    period="${3:-1y}"
    market="${4:-us}"
    curl -s "$API_BASE/api/history?symbol=$symbol&period=$period&market=$market"
    ;;
  backtest)
    symbol="$2"
    strategy="${3:-sma_crossover}"
    period="${4:-1y}"
    market="${5:-us}"
    curl -s -X POST "$API_BASE/api/backtest" \
      -H "Content-Type: application/json" \
      -d "{\"symbol\":\"$symbol\",\"market\":\"$market\",\"period\":\"$period\",\"strategy\":\"$strategy\",\"fast_period\":10,\"slow_period\":30}"
    ;;
  etf-compare)
    symbols="$2"
    period="${3:-5y}"
    market="${4:-us}"
    curl -s -X POST "$API_BASE/api/etf/compare" \
      -H "Content-Type: application/json" \
      -d "{\"symbols\":[$symbols],\"period\":\"$period\",\"market\":\"$market\"}"
    ;;
  news-us)
    curl -s "$API_BASE/api/news/us"
    ;;
  news-cn)
    curl -s "$API_BASE/api/news/cn"
    ;;
  news-hk)
    curl -s "$API_BASE/api/news/hk"
    ;;
  undervalued)
    top_n="${2:-10}"
    curl -s "$API_BASE/api/etf/undervalued?top_n=$top_n"
    ;;
  health)
    curl -s "$API_BASE/api/health"
    ;;
  *)
    echo "Usage: $0 {history|backtest|etf-compare|news-us|news-cn|news-hk|undervalued|health} [args...]"
    exit 1
    ;;
esac