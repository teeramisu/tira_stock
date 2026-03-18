#!/bin/bash
# Lobster Stock Reporter - 定期向用户推送股票数据
# 使用方法: ./lobster_report.sh [report_type]
# report_type: daily, weekly, news, watchlist

TELEGRAM_BOT_TOKEN="8782713027:AAEd4rXwDOapuIEtPWD_lgMyQFjrfsXDtR4"
CHAT_ID="${TELEGRAM_CHAT_ID}"  # 需要设置环境变量或手动指定
API_BASE="http://localhost:8000"

send_message() {
    local message="$1"
    if [ -n "$CHAT_ID" ]; then
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d "chat_id=$CHAT_ID" \
            -d "text=$message" \
            -d "parse_mode=Markdown"
    else
        echo "CHAT_ID not set, message not sent: $message"
    fi
}

case "$1" in
    daily)
        message="🦞 *每日市场简报*

获取最新市场数据..."
        
        # 获取美股新闻
        news=$(curl -s "$API_BASE/api/news/us" | jq -r '.items[:3] | .[] | "• \(.title)"' 2>/dev/null || echo "获取新闻失败")
        
        message="$message

*最新美股新闻:*
$news

使用 /lobstock 获取更多功能"
        send_message "$message"
        ;;
    weekly)
        # 获取低估ETF
        etfs=$(curl -s "$API_BASE/api/etf/undervalued?top_n=5")
        
        message="🦞 *每周ETF推荐*

$etfs"
        send_message "$message"
        ;;
    watchlist)
        # 用户自选股更新（需要配置watchlist）
        symbols="$2"
        if [ -z "$symbols" ]; then
            symbols="AAPL,MSFT,GOOGL"
        fi
        
        for symbol in $(echo "$symbols" | tr ',' ' '); do
            data=$(curl -s "$API_BASE/api/history?symbol=$symbol&period=1mo&market=us")
            echo "$data"
        done
        ;;
    *)
        echo "Usage: $0 {daily|weekly|watchlist}"
        exit 1
        ;;
esac