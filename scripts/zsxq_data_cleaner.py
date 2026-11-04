#!/usr/bin/env python3
"""知识星球(zsxq)股票推荐数据清洗 CLI。

用法:
    python scripts/zsxq_data_cleaner.py            # 规则分（不调用大模型，秒级）
    python scripts/zsxq_data_cleaner.py --llm      # 规则分 + DeepSeek 精评 Top60
    python scripts/zsxq_data_cleaner.py --llm --llm-top-n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

# 允许通过 .env 提供 DEEPSEEK_API_KEY
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from services.stock_recommendation import (  # noqa: E402
    CLEANED_FILE,
    CSV_FILE,
    SCORED_FILE,
    score_all_recommendations,
)
from services.zsxq_parser import load_topics  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="清洗知识星球股票推荐数据")
    parser.add_argument("--llm", action="store_true", help="调用 DeepSeek 大模型精评")
    parser.add_argument("--llm-top-n", type=int, default=120, help="送入大模型评分的候选数")
    parser.add_argument("--top", type=int, default=10, help="打印前 N 只")
    args = parser.parse_args()

    topics = load_topics()
    print(f"读取帖子: {len(topics)} 条（多个语料文件已按 topic_id 去重）")

    result = score_all_recommendations(topics=topics, use_llm=args.llm, llm_top_n=args.llm_top_n)

    print(f"提取股票: {result['total_stocks']} 只（可交易 {result['listed_count']} 只）")
    print(f"推荐提及: {result['total_mentions']} 次")
    print(f"大模型评分: {'是 · ' + result['llm_model'] if result['llm_used'] else '否（规则分）'}")
    print(f"行情截至: {result.get('data_as_of') or '无本地行情'}")
    print(f"输出: {CLEANED_FILE.name}, {SCORED_FILE.name}, {CSV_FILE.name}")

    print(f"\n综合评分 Top {args.top}:")
    for i, s in enumerate(result["stocks"][: args.top], 1):
        perf = s.get("performance") or {}
        print(
            f"{i:>2}. {s['stock_name']}({s['stock_code']}) "
            f"评分 {s['comprehensive_score']:>5} | {s['industry']:<8} "
            f"| 来源 {s['source_count']:>2} | 周 {s['expected_weekly']:+.1f}% "
            f"| 月 {s['expected_monthly']:+.1f}% "
            f"| 近3月 {perf.get('return_60d_pct', 'N/A')}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
