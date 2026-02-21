"""市场新闻聚合：美股 & A 股相关新闻（仅标题 + 链接）。

说明：
- 优先使用「东方财富网」作为新闻源（更专业、过滤更严格）。
- 备选「新浪财经」，但会严格过滤导航链接和无关内容。
- 这些新闻大多转载自路透社、彭博等机构或交易所公告，真实性与时效性较好。
"""
from __future__ import annotations

from typing import Any

import re

import requests

# 东方财富网：更专业的财经新闻源
EASTMONEY_US_NEWS = "http://finance.eastmoney.com/news/usstock.html"
EASTMONEY_CN_NEWS = "http://finance.eastmoney.com/news/cjxw.html"
EASTMONEY_FUTURES_NEWS = "http://finance.eastmoney.com/news/qhxw.html"
EASTMONEY_HK_NEWS = "http://finance.eastmoney.com/news/hkstock.html"

# 新浪财经：备选源
SINA_US_NEWS = "https://finance.sina.com.cn/stock/usstock/"
SINA_CN_NEWS = "https://finance.sina.com.cn/stock/"
SINA_FUTURES_NEWS = "https://finance.sina.com.cn/futures/"
SINA_HK_NEWS = "https://finance.sina.com.cn/stock/hkstock/"


def _clean_html(text: str) -> str:
    """移除 HTML 标签。"""
    return re.sub(r"<[^>]+>", "", text).strip()


def _fix_mojibake(text: str) -> str:
    """
    修复常见的 UTF-8 -> Latin1 乱码，例如 “ä¼ä¸å®¶”.
    若修复失败则返回原文。
    """
    if not text:
        return text
    try:
        return text.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _is_valid_news_link(url: str, title: str) -> bool:
    """
    判断链接是否为真正的新闻文章（而非导航、首页等）。
    """
    if not url or not title:
        return False
    
    # 过滤导航关键词
    nav_keywords = [
        "首页", "行情", "滚动", "涨幅榜", "跌幅榜", "成交额", "换手率",
        "资金流向", "沪港通", "环球指数", "重要性", "title=",
        "财经首页", "行情中心", "美股滚动", "重要性：",
    ]
    title_lower = title.lower()
    for kw in nav_keywords:
        if kw in title or kw.lower() in title_lower:
            return False
    
    # 新浪：只接受包含日期或文章ID的URL（新闻详情页）
    if "sina.com.cn" in url:
        # 格式如：/stock/usstock/c/2026-02-09/doc-xxx.shtml 或 /roll/xxx.html
        if re.search(r'/\d{4}-\d{2}-\d{2}/|/doc-|/roll/', url):
            return True
        return False
    
    # 东方财富：通常包含文章ID
    if "eastmoney.com" in url:
        # 格式如：/a/20260209xxx.html 或 /news/xxx.html
        if re.search(r'/a/\d+|/news/', url):
            return True
        return False
    
    return True


def _filter_title(title: str) -> bool:
    """过滤明显不是新闻的标题。"""
    if not title or len(title) < 4:
        return False
    
    # 过滤导航、广告类标题
    invalid_patterns = [
        r"^[A-Za-z\s]+$",  # 纯英文（可能是导航）
        r"^[\d\s]+$",  # 纯数字
        r"^[^\u4e00-\u9fa5]{0,2}$",  # 几乎没有中文
    ]
    for pattern in invalid_patterns:
        if re.match(pattern, title):
            return False
    
    return True


def _detect_encoding_and_text(resp: requests.Response) -> str:
    """根据响应自动选择合适编码并返回 text。"""
    if not resp.encoding:
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def _fetch_summary(url: str, timeout: float = 4.0, max_len: int = 80) -> str:
    """
    尝试抓取新闻正文的首段作为摘要。
    为控制性能，仅在少量新闻上调用，失败则返回空串。
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        html = _detect_encoding_and_text(resp)
    except Exception:
        return ""

    # 简单匹配正文中的 <p> 段落
    p_pattern = re.compile(r"<p[^>]*>(?P<text>.*?)</p>", re.S | re.I)
    for m in p_pattern.finditer(html):
        raw = m.group("text")
        txt = _clean_html(raw)
        txt = _fix_mojibake(txt)
        # 过滤过短或明显为版权/广告的段落
        if len(txt) < 12:
            continue
        if "版权所有" in txt or "新浪声明" in txt:
            continue
        return txt[:max_len]
    return ""


def _fetch_eastmoney_news(url: str, market: str, limit: int) -> list[dict[str, Any]]:
    """抓取东方财富网新闻。"""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = _detect_encoding_and_text(resp)
    except Exception:
        return []
    
    # 东方财富新闻链接格式：<a href="/a/20260209xxx.html" 或完整URL
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>(?:https?://finance\.eastmoney\.com)?[^"]+)"[^>]*>(?P<title>[^<]{8,})</a>',
        re.S,
    )
    items: list[dict[str, Any]] = []
    seen_links: set[str] = set()
    with_summary_budget = 5
    
    for m in pattern.finditer(html):
        href = m.group("href")
        if not href.startswith("http"):
            href = "http://finance.eastmoney.com" + href
        # 确保是finance.eastmoney.com域名
        if "finance.eastmoney.com" not in href:
            continue
        raw_title = m.group("title")
        title = _fix_mojibake(_clean_html(raw_title))
        
        if not _filter_title(title) or not _is_valid_news_link(href, title):
            continue
        if href in seen_links:
            continue
        
        seen_links.add(href)
        summary = ""
        if with_summary_budget > 0:
            summary = _fetch_summary(href)
            with_summary_budget -= 1
        
        items.append({
            "title": title,
            "link": href,
            "published_at": "",
            "source": "EastMoney",
            "summary": summary,
            "market": market,
        })
        
        if len(items) >= limit:
            break
    
    return items


def get_us_stock_news(limit: int = 10) -> list[dict[str, Any]]:
    """美股相关新闻：优先东方财富，备选新浪财经。"""
    # 优先尝试东方财富
    items = _fetch_eastmoney_news(EASTMONEY_US_NEWS, "us", limit)
    if len(items) >= limit:
        return items[:limit]
    
    # 备选：新浪财经
    try:
        resp = requests.get(SINA_US_NEWS, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = _detect_encoding_and_text(resp)
    except Exception:
        return items
    
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>https?://finance\.sina\.com\.cn[^"]+)"[^>]*>(?P<title>[^<]{8,})</a>',
        re.S,
    )
    seen_links = {item["link"] for item in items}
    with_summary_budget = max(0, 5 - len(items))
    
    bad_cn_keywords = [
        "黄金", "白银", "期市", "期货", "外汇", "贵金属", "原油",
        "机构看盘", "机构看市", "机构看盘：", "机构看市：",
        "滚动新闻", "财经首页", "期市要闻",
        "机器人送礼", "赛博年货",
    ]

    for m in pattern.finditer(html):
        href = m.group("href")
        raw_title = m.group("title")
        title = _fix_mojibake(_clean_html(raw_title))

        # 过滤明显不是 A 股新闻的标题（期货 / 黄金 / 导航等）
        if any(kw in title for kw in bad_cn_keywords):
            continue
        if not _filter_title(title) or not _is_valid_news_link(href, title):
            continue
        if href in seen_links:
            continue
        
        seen_links.add(href)
        summary = ""
        if with_summary_budget > 0:
            summary = _fetch_summary(href)
            with_summary_budget -= 1
        
        items.append({
            "title": title,
            "link": href,
            "published_at": "",
            "source": "Sina Finance",
            "summary": summary,
            "market": "us",
        })
        
        if len(items) >= limit:
            break
    
    return items[:limit]


def get_cn_a_share_news(limit: int = 10) -> list[dict[str, Any]]:
    """A 股相关新闻：优先东方财富，备选新浪财经。"""
    # 优先尝试东方财富
    items = _fetch_eastmoney_news(EASTMONEY_CN_NEWS, "cn", limit)
    if len(items) >= limit:
        return items[:limit]
    
    # 备选：新浪财经
    try:
        resp = requests.get(SINA_CN_NEWS, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = _detect_encoding_and_text(resp)
    except Exception:
        return items
    
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>https?://finance\.sina\.com\.cn[^"]+)"[^>]*>(?P<title>[^<]{8,})</a>',
        re.S,
    )
    seen_links = {item["link"] for item in items}
    with_summary_budget = max(0, 5 - len(items))
    
    for m in pattern.finditer(html):
        href = m.group("href")
        raw_title = m.group("title")
        title = _fix_mojibake(_clean_html(raw_title))
        
        if not _filter_title(title) or not _is_valid_news_link(href, title):
            continue
        if href in seen_links:
            continue
        
        seen_links.add(href)
        summary = ""
        if with_summary_budget > 0:
            summary = _fetch_summary(href)
            with_summary_budget -= 1
        
        items.append({
            "title": title,
            "link": href,
            "published_at": "",
            "source": "Sina Finance",
            "summary": summary,
            "market": "cn",
        })
        
        if len(items) >= limit:
            break
    
    return items[:limit]


def get_futures_news(limit: int = 10) -> list[dict[str, Any]]:
    """期货相关新闻：优先东方财富，备选新浪财经。"""
    # 优先尝试东方财富
    items = _fetch_eastmoney_news(EASTMONEY_FUTURES_NEWS, "futures", limit)
    if len(items) >= limit:
        return items[:limit]

    # 备选：新浪财经
    try:
        resp = requests.get(SINA_FUTURES_NEWS, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = _detect_encoding_and_text(resp)
    except Exception:
        return items

    pattern = re.compile(
        r'<a[^>]+href="(?P<href>https?://finance\.sina\.com\.cn[^"]+)"[^>]*>(?P<title>[^<]{8,})</a>',
        re.S,
    )
    seen_links = {item["link"] for item in items}
    with_summary_budget = max(0, 5 - len(items))

    for m in pattern.finditer(html):
        href = m.group("href")
        raw_title = m.group("title")
        title = _fix_mojibake(_clean_html(raw_title))

        if not _filter_title(title) or not _is_valid_news_link(href, title):
            continue
        if href in seen_links:
            continue

        seen_links.add(href)
        summary = ""
        if with_summary_budget > 0:
            summary = _fetch_summary(href)
            with_summary_budget -= 1

        items.append(
            {
                "title": title,
                "link": href,
                "published_at": "",
                "source": "Sina Finance",
                "summary": summary,
                "market": "futures",
            }
        )

        if len(items) >= limit:
            break

    return items[:limit]


def get_hk_stock_news(limit: int = 10) -> list[dict[str, Any]]:
    """港股相关新闻：优先东方财富，备选新浪财经。"""
    # 优先尝试东方财富
    items = _fetch_eastmoney_news(EASTMONEY_HK_NEWS, "hk", limit)
    if len(items) >= limit:
        return items[:limit]

    # 备选：新浪财经
    try:
        resp = requests.get(SINA_HK_NEWS, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = _detect_encoding_and_text(resp)
    except Exception:
        return items

    pattern = re.compile(
        r'<a[^>]+href="(?P<href>https?://finance\.sina\.com\.cn[^"]+)"[^>]*>(?P<title>[^<]{8,})</a>',
        re.S,
    )
    seen_links = {item["link"] for item in items}
    with_summary_budget = max(0, 5 - len(items))

    for m in pattern.finditer(html):
        href = m.group("href")
        raw_title = m.group("title")
        title = _fix_mojibake(_clean_html(raw_title))

        if not _filter_title(title) or not _is_valid_news_link(href, title):
            continue
        if href in seen_links:
            continue

        seen_links.add(href)
        summary = ""
        if with_summary_budget > 0:
            summary = _fetch_summary(href)
            with_summary_budget -= 1

        items.append(
            {
                "title": title,
                "link": href,
                "published_at": "",
                "source": "Sina Finance",
                "summary": summary,
                "market": "hk",
            }
        )

        if len(items) >= limit:
            break

    return items[:limit]
