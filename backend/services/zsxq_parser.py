"""知识星球(zsxq)帖子解析：从非结构化帖子中提取结构化股票推荐。

设计要点
- 语料：扫描 zsxq_output/*.json 与 crawler/data/zsxq/topics_raw*.json，按 topic_id 去重。
- 标的识别：以本地证券主数据（DuckDB instruments）与板块成份股（board_constituents.json）
  的股票名称为词典做最长匹配，避免硬编码映射导致的错码/漏码。
- 板块：使用同花顺行业（industry）+ 概念（concept），不再用全文关键词猜测。
- 聚合：按股票代码聚合，source_count = 提及该股的**不同帖子数**。
"""
from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BOARD_FILE = DATA_DIR / "board_constituents.json"
MARKET_DB = DATA_DIR / "market" / "market_data.duckdb"

# 语料文件（按优先级，重复 topic_id 只取一次）
TOPIC_FILES = [
    PROJECT_ROOT.parent / "zsxq_output" / "scrolled_topics.json",
    PROJECT_ROOT.parent / "zsxq_output" / "all_topics.json",
    PROJECT_ROOT.parent / "zsxq_output" / "latest_topics.json",
    PROJECT_ROOT.parent / "zsxq_output" / "topics.json",
    PROJECT_ROOT.parent / "crawler" / "data" / "zsxq" / "topics_raw.json",
    PROJECT_ROOT.parent / "crawler" / "data" / "zsxq" / "topics_raw2.json",
    PROJECT_ROOT.parent / "crawler" / "data" / "zsxq" / "topics_raw3.json",
]

# 推荐语义信号：命中才认为帖子在“推荐/关注”，降低把行业名当个股的误报
RECOMMEND_HINTS = (
    "推荐", "关注", "建议", "看好", "标的", "受益", "买入", "增持", "布局",
    "涨停", "龙头", "首推", "重点", "超预期", "低估", "弹性", "催化", "机会",
)
# 强推荐信号（用于计算推荐强度）
STRONG_HINTS = ("重点推荐", "首推", "强烈推荐", "买入", "增持", "坚定看好", "持续推荐")

# 通用词，避免把行情描述当个股
NAME_STOPLIST = {
    "市场", "板块", "行业", "公司", "股票", "指数", "基金", "龙头", "概念",
    "机器人", "人工智能", "大数据", "云计算", "新能源", "半导体", "军工",
    "银行", "保险", "证券", "地产", "医药", "消费", "通信", "电子",
}

EMOJI_TOKEN = re.compile(r"\[[^\[\]\s]{1,6}\]")
TAG_RE = re.compile(r"<e\s+[^>]*/>|<[^>]+>")
TAG_TITLE_RE = re.compile(r"<e\s+[^>]*?title=\"([^\"]*)\"[^>]*/>")
SENT_SPLIT = re.compile(r"[。！？；\n\r]+")


@dataclass
class Mention:
    name: str
    code: Optional[str]
    reason: str


@dataclass
class StockRecord:
    code: str
    name: str
    industry: str = "未分类"
    concepts: List[str] = field(default_factory=list)
    sources: List[dict] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    likes: int = 0
    readers: int = 0
    strength: float = 0.0
    strong_count: int = 0
    first_seen: str = ""
    last_seen: str = ""

    @property
    def listed(self) -> bool:
        return bool(re.fullmatch(r"\d{6}", self.code or ""))

    def to_dict(self) -> dict:
        # 前端兼容字段：sectors = 行业 + 主题概念
        sectors = [self.industry] if self.industry and self.industry != "未分类" else []
        sectors += self.concepts[:4]
        if not sectors:
            sectors = ["未分类"]
        return {
            "stock_name": self.name,
            "stock_code": self.code or "未知",
            "code": self.code or "",
            "industry": self.industry,
            "concepts": self.concepts,
            "sectors": sectors,
            "listed": self.listed,
            "source_count": len(self.sources),
            "author_count": len({s.get("author", "") for s in self.sources}),
            "strong_count": self.strong_count,
            "strength": round(self.strength, 2),
            "likes": self.likes,
            "readers": self.readers,
            "sources": self.sources[:10],
            "reasons": self.reasons[:5],
            "reasons_summary": "；".join(self.reasons[:3]),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


# ---------------------------------------------------------------- 语料读取
def load_topics(paths: Optional[Iterable[Path]] = None) -> List[dict]:
    """读取所有语料并按 topic_id 去重（按给定顺序优先）。"""
    files = list(paths) if paths else TOPIC_FILES
    seen: set = set()
    out: List[dict] = []
    for path in files:
        path = Path(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for topic in _iter_topics(data):
            tid = str(topic.get("topic_id") or topic.get("topic_uid") or "")
            if not tid or tid in seen:
                continue
            seen.add(tid)
            out.append(topic)
    return out


def _iter_topics(data: Any) -> Iterable[dict]:
    if isinstance(data, list):
        yield from (x for x in data if isinstance(x, dict))
        return
    if isinstance(data, dict):
        if isinstance(data.get("topics"), list):
            yield from (x for x in data["topics"] if isinstance(x, dict))
            return
        resp = data.get("resp_data")
        if isinstance(resp, dict) and isinstance(resp.get("topics"), list):
            yield from (x for x in resp["topics"] if isinstance(x, dict))


def topic_text(topic: dict) -> str:
    talk = topic.get("talk") or topic.get("article") or topic.get("question") or {}
    if isinstance(talk, dict):
        return talk.get("text") or talk.get("title") or ""
    return ""


def topic_author(topic: dict) -> str:
    talk = topic.get("talk") or {}
    owner = (talk.get("owner") if isinstance(talk, dict) else None) or {}
    return owner.get("name") or ""


def clean_text(raw: str) -> str:
    """去掉知识星球富文本标签、URL 编码标题、表情标记。"""
    if not raw:
        return ""
    text = raw

    def _title(m: re.Match) -> str:
        try:
            return urllib.parse.unquote(m.group(1)) or ""
        except Exception:
            return ""

    text = TAG_TITLE_RE.sub(_title, text)
    text = TAG_RE.sub(" ", text)
    text = EMOJI_TOKEN.sub(" ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- 标的词典
class StockIndex:
    """股票名称 → 代码 / 行业 / 概念 的本地词典。"""

    def __init__(self, board_file: Path = BOARD_FILE, market_db: Path = MARKET_DB):
        self.name_to_codes: Dict[str, List[str]] = {}
        self.code_to_name: Dict[str, str] = {}
        self.name_to_industry: Dict[str, str] = {}
        self.name_to_concepts: Dict[str, List[str]] = {}
        self.code_to_industry: Dict[str, str] = {}
        self.code_to_concepts: Dict[str, List[str]] = {}
        self.concept_members: Dict[str, List[Tuple[str, Optional[str]]]] = {}
        self._first_char: Dict[str, List[str]] = {}
        self._load_board(board_file)
        self._load_instruments(market_db)
        self._build_matcher()

    def _add_name(self, name: str, code: Optional[str]) -> None:
        name = (name or "").strip()
        if len(name) < 2 or name in NAME_STOPLIST:
            return
        codes = self.name_to_codes.setdefault(name, [])
        if code and code not in codes:
            codes.append(code)
        if code and code not in self.code_to_name:
            self.code_to_name[code] = name

    def _load_board(self, board_file: Path) -> None:
        if not board_file.exists():
            return
        try:
            data = json.loads(board_file.read_text(encoding="utf-8"))
        except Exception:
            return
        for industry, payload in (data.get("industry") or {}).items():
            for item in payload.get("stocks", []):
                name, code = item.get("name"), item.get("code")
                self._add_name(name, code)
                self.name_to_industry.setdefault(name, industry)
                if code:
                    self.code_to_industry.setdefault(code, industry)
        for concept, payload in (data.get("concept") or {}).items():
            members: List[Tuple[str, Optional[str]]] = []
            for item in payload.get("stocks", []):
                name, code = item.get("name"), item.get("code")
                self._add_name(name, code)
                self.name_to_concepts.setdefault(name, []).append(concept)
                if code:
                    self.code_to_concepts.setdefault(code, []).append(concept)
                members.append((name, code))
            self.concept_members[concept] = members

    def _load_instruments(self, market_db: Path) -> None:
        if not market_db.exists():
            return
        try:
            import duckdb

            try:
                con = duckdb.connect(str(market_db), read_only=True)
            except Exception:
                # 同进程已有读写连接时，DuckDB 要求配置一致，退回读写连接（只执行 SELECT）
                con = duckdb.connect(str(market_db))
            try:
                rows = con.execute(
                    "select symbol, name, is_st from instruments "
                    "where market='cn' and asset_type='stock' and is_active "
                    "and coalesce(source, '') <> 'fixture'"
                ).fetchall()
            finally:
                con.close()
        except Exception:
            return
        for symbol, name, is_st in rows:
            # 只接受 6 位 A 股代码，过滤测试用 fixture（AAA/BBB 等）
            if not re.fullmatch(r"\d{6}", str(symbol or "")):
                continue
            self._add_name(name, symbol)

    def _build_matcher(self) -> None:
        for name in sorted(self.name_to_codes, key=len, reverse=True):
            self._first_char.setdefault(name[0], []).append(name)

    # -- 查询
    def find(self, text: str, require_hint: bool = False) -> List[Tuple[str, Optional[str]]]:
        """在文本中做最长匹配，返回 [(name, code)]，同一 code 只返回一次。"""
        if not text:
            return []
        has_hint = (not require_hint) or any(h in text for h in RECOMMEND_HINTS)
        taken: List[Tuple[int, int]] = []
        found: List[Tuple[str, Optional[str]]] = []
        found_codes: set = set()
        for i, ch in enumerate(text):
            for name in self._first_char.get(ch, ()):  # 已按长度降序
                j = i + len(name)
                if j > len(text) or not text.startswith(name, i):
                    continue
                if any(i < b and j > a for a, b in taken):  # 与更长匹配重叠
                    continue
                # 2 字名称容易误报：必须出现在 【】 内或文本带推荐语义
                if len(name) <= 2 and not (has_hint or _in_brackets(text, i, j)):
                    continue
                taken.append((i, j))
                code = (self.name_to_codes.get(name) or [None])[0]
                key = code or name
                if key in found_codes:
                    continue
                found_codes.add(key)
                found.append((name, code))
                break
        return found

    def industry(self, name: str, code: Optional[str]) -> str:
        if code and code in self.code_to_industry:
            return self.code_to_industry[code]
        if name in self.name_to_industry:
            return self.name_to_industry[name]
        return self._infer_industry(name, code)

    def _infer_industry(self, name: str, code: Optional[str]) -> str:
        """行业快照未覆盖时，用概念成份股的行业分布投票推断。"""
        concepts = self.concepts(name, code)
        votes: Dict[str, float] = {}
        for concept in concepts[:6]:
            members = self.concept_members.get(concept, [])
            if not members:
                continue
            local: Dict[str, int] = {}
            for m_name, m_code in members:
                ind = self.code_to_industry.get(m_code or "") or self.name_to_industry.get(m_name, "")
                if ind:
                    local[ind] = local.get(ind, 0) + 1
            total = sum(local.values())
            if not total:
                continue
            for ind, cnt in local.items():
                votes[ind] = votes.get(ind, 0.0) + cnt / total
        if not votes:
            return "未分类"
        best = max(votes.items(), key=lambda kv: kv[1])
        return best[0] if best[1] >= 0.6 else "未分类"

    def concepts(self, name: str, code: Optional[str]) -> List[str]:
        items = []
        if code:
            items = self.code_to_concepts.get(code, [])
        if not items:
            items = self.name_to_concepts.get(name, [])
        return list(items)


def _in_brackets(text: str, start: int, end: int) -> bool:
    left = text.rfind("【", max(0, start - 2), start + 1)
    right = text.find("】", end - 1, end + 2)
    return left != -1 and right != -1 and left < start and right >= end


def extract_mentions(
    text: str,
    index: StockIndex,
    require_hint: bool = True,
) -> List[Tuple[str, Optional[str], str]]:
    """提取文本中的股票提及，附带推荐理由句子。

    require_hint=True 时只保留含“推荐/关注/看好…”等语义的帖子，过滤纯行情播报。
    未在本地证券词典中的【】名称（多为“天风电新”这类券商团队署名）直接丢弃。
    """
    clean = clean_text(text)
    if not clean:
        return []
    if require_hint and not any(h in clean for h in RECOMMEND_HINTS):
        return []
    names = index.find(clean)
    if not names:
        return []

    sentences = [s.strip() for s in SENT_SPLIT.split(clean) if s.strip()]
    out: List[Tuple[str, Optional[str], str]] = []
    for name, code in names:
        idxs = [i for i, s in enumerate(sentences) if name in s]
        if not idxs:
            idxs = [0]
        first = idxs[0]
        start = first
        # 补上下文：上一句若是短标题/“推荐：”这类引导句，一并带上
        if first > 0:
            prev = sentences[first - 1]
            if len(prev) <= 40 or prev.endswith(("：", ":")):
                start = first - 1
        # 名称常出现在标题/名单句中，向后补 2 句，保证理由包含逻辑而不只是名字
        picked = sentences[start: first + 3]
        reason = "；".join(dict.fromkeys(s for s in picked if s))
        out.append((name, code, reason[:300]))
    return out


# ---------------------------------------------------------------- 聚合
def build_stock_dataset(
    topics: Optional[List[dict]] = None,
    index: Optional[StockIndex] = None,
    require_hint: bool = True,
) -> Dict[str, Any]:
    """解析语料并聚合为结构化股票推荐数据。"""
    topics = topics if topics is not None else load_topics()
    index = index or StockIndex()
    records: Dict[str, StockRecord] = {}
    unlisted: Dict[str, int] = {}

    for topic in topics:
        text = topic_text(topic)
        if not text:
            continue
        author = topic_author(topic)
        created = topic.get("create_time") or ""
        likes = int(topic.get("likes_count") or 0)
        readers = int(topic.get("readers_count") or 0)
        group = ""
        g = topic.get("group")
        if isinstance(g, dict):
            group = g.get("name") or ""
        tid = str(topic.get("topic_id") or topic.get("topic_uid") or "")
        strength = float(sum(text.count(h) for h in STRONG_HINTS))
        strong_hit = any(h in text for h in STRONG_HINTS)

        for name, code, reason in extract_mentions(text, index, require_hint=require_hint):
            if not code:
                unlisted[name] = unlisted.get(name, 0) + 1
                continue
            rec = records.get(code)
            if rec is None:
                rec = StockRecord(
                    code=code,
                    name=index.code_to_name.get(code, name),
                    industry=index.industry(name, code),
                    concepts=index.concepts(name, code)[:6],
                    first_seen=created,
                )
                records[code] = rec
            if not rec.first_seen or created < rec.first_seen:
                rec.first_seen = created
            if not rec.last_seen or created > rec.last_seen:
                rec.last_seen = created
            if any(s.get("topic_id") == tid for s in rec.sources):
                continue
            rec.sources.append({
                "topic_id": tid,
                "author": author,
                "group": group,
                "time": created,
                "likes": likes,
                "strength": strength,
            })
            rec.likes += likes
            rec.readers += readers
            rec.strength += strength
            if strong_hit:
                rec.strong_count += 1
            if reason and reason not in rec.reasons:
                rec.reasons.append(reason)

    stocks = [r.to_dict() for r in records.values()]
    stocks.sort(key=lambda s: (s["strong_count"], s["source_count"], s["likes"]), reverse=True)
    return {
        "total_topics": len(topics),
        "total_mentions": sum(len(r.sources) for r in records.values()),
        "unique_stocks_count": len(stocks),
        "listed_count": sum(1 for s in stocks if s["listed"]),
        "unlisted_mentions": [
            {"name": n, "count": c}
            for n, c in sorted(unlisted.items(), key=lambda kv: kv[1], reverse=True)[:50]
        ],
        "stocks": stocks,
    }
