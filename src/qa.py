"""情报问答：检索 + 回答组装（RAG 的检索层与生成层）。

从 dashboard.py 抽出来单独成模块，原因有两个：

1. **中文检索不能靠空格分词**。原实现用 ``question.split()`` 切词，中文问句
   会被切成一个整串（"2026国庆电影票房"），再拿去做 ``LIKE '%整串%'``，
   命中率几乎为零——这正是线上"库内未检索到相关情报"的直接原因。
   这里改用「中文 n-gram + 英文/数字词 + 停用词过滤」，无需引入 jieba 之类的
   分词依赖，几百到几万条量级下毫秒级完成。

2. **检索结果要能直接当答案用**。未配置 LLM 时不再是丢一串链接，而是给出
   命中概览（条数/时间跨度/来源分布/情感分布）+ 逐条要点摘要；配置 LLM 后
   升级为引用编号的自然语言回答（[1][2] 与来源列表一一对应）。

本模块不依赖 Streamlit，便于单元测试；对外主入口是 :func:`answer_question`。
"""

from __future__ import annotations

import json
import re
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime

import numpy as np

from .rag import _from_blob, cosine

# --------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------

_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_ASCII_RE = re.compile(r"[a-z0-9][a-z0-9._+-]*")

# 停用词：疑问词、语气词、时间泛词与"动态/资讯"这类对检索无区分度的词。
# 不剔除的话，"这周有什么动态" 会带来一堆命中所有文章的噪声词。
STOPWORDS = {
    "的", "了", "是", "在", "有", "和", "与", "及", "或", "也", "都", "就", "被",
    "这", "那", "哪", "哪些", "哪个", "什么", "怎么", "怎样", "如何", "为何",
    "为什么", "请问", "请教", "一下", "可以", "能否", "是否", "有没有", "多少",
    "最近", "近期", "目前", "现在", "今天", "今日", "昨日", "明天", "这周", "本周",
    "上周", "上个月", "这个月", "今年", "去年", "动态", "情况", "消息", "新闻",
    "资讯", "情报", "行业", "新品", "发布", "推出", "上市", "开售", "官宣", "公布",
    "宣布", "正式", "近日", "值得", "关注", "关于", "对于", "以及", "我们", "你们", "他们",
    "介绍", "分析", "总结", "讲讲", "说说", "看看", "告诉", "知道", "了解", "发生",
    "出现", "进展", "表现", "方面", "相关", "主要", "重要", "影响", "怎么样", "怎样呢",
    "吗", "呢", "吧", "啊", "嘛", "哦", "呀", "de", "the", "a", "an", "of", "and",
    "is", "are", "what", "how", "why", "please",
}

# 同义扩展：口语说法 / 简写与库内常见写法互认，提升召回。
_ALIASES = {
    "双11": ("双十一",),
    "双十一": ("双11",),
    "双12": ("双十二",),
    "gmv": ("成交额", "交易额"),
    "ai": ("人工智能", "大模型"),
    "财报": ("业绩", "季报", "年报"),
    "裁员": ("人员优化", "组织调整"),
    "低价": ("价格战", "补贴"),
    "外卖": ("即时零售", "本地生活"),
}

_MAX_TOKENS = 40


def tokenize(text: str) -> list:
    """把问题切成检索词：英文/数字按词、中文按 1~3 字 n-gram，去停用词。

    中文没有空格，n-gram 是最省依赖的通用做法：
    "2026国庆电影票房" → ["2026", "国庆", "庆电", "电影", "影票", "票房", ...]，
    其中 "国庆/电影/票房" 能命中库内相关报道，而 "庆电" 这类碎片因为查不到
    任何文档自然得 0 分，不影响排序。
    """
    text = (text or "").lower()
    tokens = []

    for m in _ASCII_RE.finditer(text):
        tok = m.group(0).strip("._+-")
        if not tok:
            continue
        if tok.isdigit() and len(tok) < 4:
            continue  # 单双位数字（如年份以外的序数）区分度太低
        tokens.append(tok)

    for m in _CJK_RE.finditer(text):
        run = m.group(0)
        if len(run) <= 4:
            tokens.append(run)  # 短词整体保留："京东"、"拼多多"
        for n in (2, 3):
            if len(run) >= n:
                tokens.extend(run[i : i + n] for i in range(len(run) - n + 1))
        if len(run) == 1:
            tokens.append(run)

    out, seen = [], set()
    for t in tokens:
        if not t or t in STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= _MAX_TOKENS:
            break
    return out


def expand_tokens(tokens: list) -> list:
    """加入同义词，保持原词在前（原词权重更高）。"""
    out = list(tokens)
    for t in tokens:
        for syn in _ALIASES.get(t, ()):
            if syn not in out:
                out.append(syn)
    return out


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


class CallBudget:
    """LLM 调用的滑动窗口预算。

    看板部署在公网且没有登录，任何访客提问都会消耗账号额度。这里对"生成式回答"
    设每小时上限，超限不报错、只是退回抽取式回答——功能不缺失，额度也不会被刷穿。
    用 monotonic 时钟，避免系统时间被调整时窗口计算错乱。
    """

    def __init__(self, max_per_window: int = 60, window_seconds: float = 3600.0, clock=time.monotonic):
        self.max_per_window = max(1, int(max_per_window))
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._calls = deque()

    def _trim(self, now: float) -> None:
        while self._calls and now - self._calls[0] > self.window_seconds:
            self._calls.popleft()

    def allow(self, now: float = None) -> bool:
        """占用一次额度；返回 False 表示本次已超限。"""
        now = self._clock() if now is None else float(now)
        self._trim(now)
        if len(self._calls) >= self.max_per_window:
            return False
        self._calls.append(now)
        return True

    def remaining(self, now: float = None) -> int:
        self._trim(self._clock() if now is None else float(now))
        return max(0, self.max_per_window - len(self._calls))

    @property
    def used(self) -> int:
        return len(self._calls)


@dataclass
class Hit:
    """一条命中的情报。score 为 0~1 的可解释匹配度（字段加权后归一）。"""

    id: int
    title: str
    url: str = ""
    source: str = ""
    published: str = ""
    fetched_at: str = ""
    summary: str = ""
    sentiment: str = ""
    tags: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    score: float = 0.0
    matched: list = field(default_factory=list)
    similarity: float = -1.0
    raw_score: float = 0.0

    @property
    def date(self) -> str:
        return normalize_date(self.published) or normalize_date(self.fetched_at)

    def as_row(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "date": self.date,
            "summary": self.summary,
            "sentiment": self.sentiment,
            "tags": self.tags,
            "entities": self.entities,
            "score": self.score,
            "matched": self.matched,
            "similarity": self.similarity,
        }


@dataclass
class QAAnswer:
    """一次问答的完整结果，UI 只需要负责渲染。"""

    answer: str
    hits: list = field(default_factory=list)
    mode: str = "keyword"          # hybrid | keyword | vector | none
    mode_note: str = ""
    stats: dict = field(default_factory=dict)
    warning: str = ""              # 向量不可用 / LLM 失败等降级提示
    suggestions: list = field(default_factory=list)
    llm_used: bool = False

    @property
    def hit_count(self) -> int:
        return len(self.hits)


# --------------------------------------------------------------------------
# 候选集与关键词排序
# --------------------------------------------------------------------------

_FIELD_WEIGHT = (("title", 3.0), ("tags", 2.0), ("entities", 2.0), ("summary", 1.2), ("source", 0.6))


def load_candidates(conn, days: int = None, limit: int = 5000) -> list:
    """读取候选情报（含 AI 分析字段与向量）。days 为空表示不限时间、全库检索。"""
    sql = """
        SELECT a.id, a.title, a.url, a.source, a.published, a.fetched_at,
               a.summary, a.embedding,
               an.summary_ai, an.sentiment, an.tags, an.entities
        FROM articles a LEFT JOIN analysis an ON an.article_id = a.id
    """
    params = []
    if days:
        sql += " WHERE date(a.fetched_at) >= date('now', ?)"
        params.append(f"-{int(days)} days")
    sql += " ORDER BY a.id DESC LIMIT ?"
    params.append(int(limit))
    return [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()]


def _join_json(value) -> str:
    if not value:
        return ""
    try:
        items = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return str(value)
    if isinstance(items, list):
        return " ".join(str(x) for x in items)
    return str(items)


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def normalize_date(value) -> str:
    """把 RSS 的 RFC822 日期（"Wed, 12 Aug 2026 10:45:21 GMT"）归一成 YYYY-MM-DD。

    RSS 源给的是 RFC822，直接截前 10 个字符会得到 "Wed, 12 Au" 这种半截字符串，
    既不能按时间排序也不适合展示。
    """
    text = str(value or "").strip()
    if not text:
        return ""
    match = _ISO_DATE_RE.search(text)
    if match and text.startswith(match.group(0)):
        return match.group(0)
    try:
        return parsedate_to_datetime(text).date().isoformat()
    except (TypeError, ValueError):
        pass
    if match:
        return match.group(0)
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return ""


def _fields(row: dict) -> dict:
    # 统一转小写：查询词已被 tokenize 小写化（"iphone"），文档侧若保留 "iPhone"
    # 就会互不匹配，出现"标题里明明有 iPhone 却排不上来"的诡异排序。
    return {
        "title": str(row.get("title") or "").lower(),
        "summary": str(row.get("summary_ai") or row.get("summary") or "").lower(),
        "tags": _join_json(row.get("tags")).lower(),
        "entities": _join_json(row.get("entities")).lower(),
        "source": str(row.get("source") or "").lower(),
    }


def score_row(tokens: list, row: dict) -> tuple:
    """按字段加权给单条情报打分，返回 (原始分, 命中词)。

    同一个词命中多个字段只取最高权重，避免"标题里出现 5 次"把分数刷爆；
    最后乘覆盖率系数，让"命中的是查询里的核心词"比"只蹭到一个碎词"排前面。
    """
    fields = _fields(row)
    score = 0.0
    matched = []
    for tok in tokens:
        best = 0.0
        for name, weight in _FIELD_WEIGHT:
            if tok in fields[name]:
                best = max(best, weight)
        if best:
            matched.append(tok)
            score += best + (0.4 if len(tok) >= 2 else 0.0)
    if matched:
        coverage = len(matched) / max(1, len(tokens))
        score *= 0.6 + 0.4 * coverage
    return score, matched


def _is_meaningful(token: str) -> bool:
    """年份、序号这类纯数字词区分度太低：整个库都是"2026"年的新闻，
    只靠它命中会把"2026国庆电影票房"答成一堆无关的 2026 年文章。"""
    return len(token) >= 2 and not token.isdigit()


def keyword_rank(rows: list, tokens: list, limit: int = 10) -> list:
    """关键词召回 + 排序。

    只有一条硬性要求：至少命中一个"有区分度"的词（长度 ≥2 且不是纯数字）。
    "只蹭到单字碎片"或"只蹭到年份"的结果不算命中，宁可走零命中引导——
    这正是线上"2026国庆电影票房"答出一堆 2026 年无关文章的原因。

    至于"只沾到一个满库通用词"（如问折叠屏却只命中"手机"）这类精度问题，
    刻意不在召回层一刀切：真库里常有"只共享一个关键词但确实相关"的情报
    （问"最近电影行业动态"而库里只有一篇讲电影银幕的报道），切掉会误伤。
    这类结果交给回答层处理——有 LLM 时由模型明确说明"资料不足"，没有 LLM 时
    按匹配度排序展示候选。
    """
    scored = []
    for row in rows:
        raw, matched = score_row(tokens, row)
        if not matched:
            continue
        if not any(_is_meaningful(t) for t in matched):
            continue
        scored.append((raw, matched, row))

    if not scored:
        return _relaxed_rank(rows, tokens, limit)

    scored.sort(key=lambda x: (-x[0], -x[2].get("id", 0)))
    top = scored[:limit]
    best = top[0][0] or 1.0
    hits = []
    for raw, matched, row in top:
        hit = _to_hit(row)
        hit.raw_score = raw
        hit.score = round(raw / best, 4)
        hit.matched = matched
        hits.append(hit)
    return hits


def _relaxed_rank(rows: list, tokens: list, limit: int = 10) -> list:
    """宽松兜底：只命中单字碎片时，按命中字数排序，并标注为弱匹配。"""
    scored = []
    for row in rows:
        raw, matched = score_row(tokens, row)
        if len(matched) >= 2:
            scored.append((raw, matched, row))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], -x[2].get("id", 0)))
    top = scored[:limit]
    best = top[0][0] or 1.0
    hits = []
    for raw, matched, row in top:
        hit = _to_hit(row)
        hit.raw_score = raw
        hit.score = round(raw / best * 0.6, 4)  # 弱匹配整体降权
        hit.matched = matched
        hits.append(hit)
    return hits


def _to_hit(row: dict) -> Hit:
    return Hit(
        id=int(row.get("id") or 0),
        title=str(row.get("title") or ""),
        url=str(row.get("url") or ""),
        source=str(row.get("source") or ""),
        published=str(row.get("published") or ""),
        fetched_at=str(row.get("fetched_at") or ""),
        summary=str(row.get("summary_ai") or row.get("summary") or ""),
        sentiment=str(row.get("sentiment") or ""),
        tags=_to_list(row.get("tags")),
        entities=_to_list(row.get("entities")),
    )


def _to_list(value) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return [str(value)]
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return [str(parsed)]


# --------------------------------------------------------------------------
# 向量检索与混合排序
# --------------------------------------------------------------------------


def vector_rank(rows: list, query_vec, limit: int = 10) -> list:
    """对已入库向量做全量余弦检索。向量维度不一致（换过模型）时返回空表。"""
    usable = [r for r in rows if r.get("embedding")]
    if not usable:
        return []
    try:
        matrix = np.stack([_from_blob(r["embedding"]) for r in usable])
    except ValueError:
        return []
    if matrix.shape[1] != len(query_vec):
        return []
    sims = cosine(np.asarray(query_vec, dtype=np.float32), matrix)
    order = np.argsort(sims)[::-1][:limit]
    hits = []
    for i in order:
        hit = _to_hit(usable[i])
        hit.similarity = round(float(sims[i]), 4)
        hit.score = round(max(0.0, float(sims[i])), 4)
        hits.append(hit)
    return hits


def hybrid_merge(keyword_hits: list, vector_hits: list, limit: int = 6, alpha: float = 0.6) -> list:
    """关键词与向量结果融合：两路都命中的排最前，只在单路出现的按加权分排序。

    alpha 是向量权重——语义召回更准，但库内数据量小、向量模型对短问句不够稳，
    因此保留 40% 的字面匹配权重，让"问什么就出什么"的可解释性不至于丢失。
    """
    if not vector_hits:
        return keyword_hits[:limit]
    if not keyword_hits:
        return vector_hits[:limit]

    kw_best = max((h.raw_score or h.score) for h in keyword_hits) or 1.0
    combined = {}
    for h in keyword_hits:
        combined[h.id] = [h, (h.raw_score or h.score) / kw_best, 0.0]
    for h in vector_hits:
        sim = max(0.0, h.similarity)
        if h.id in combined:
            combined[h.id][2] = sim
        else:
            combined[h.id] = [h, 0.0, sim]

    def fused(item):
        hit, kw, vec = item
        both = 0.15 if (kw > 0 and vec > 0) else 0.0
        return alpha * vec + (1 - alpha) * kw + both

    ranked = sorted(combined.values(), key=lambda it: (-fused(it), -it[0].id))[:limit]
    out = []
    best = fused(ranked[0]) or 1.0
    for hit, _kw, _vec in ranked:
        hit.score = round(min(1.0, fused((hit, _kw, _vec)) / best), 4)
        out.append(hit)
    return out


# --------------------------------------------------------------------------
# 回答组装
# --------------------------------------------------------------------------


def hit_stats(hits: list, scanned: int = 0) -> dict:
    """命中概览：条数、时间跨度、来源分布、情感分布、高频实体。"""
    sources = {}
    sentiments = {}
    entities = {}
    dates = []
    for h in hits:
        if h.source:
            sources[h.source] = sources.get(h.source, 0) + 1
        if h.sentiment:
            sentiments[h.sentiment] = sentiments.get(h.sentiment, 0) + 1
        for e in h.entities:
            entities[e] = entities.get(e, 0) + 1
        if h.date:
            dates.append(h.date)
    top_entities = sorted(entities.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    return {
        "hits": len(hits),
        "scanned": scanned,
        "date_from": min(dates) if dates else "",
        "date_to": max(dates) if dates else "",
        "sources": sorted(sources.items(), key=lambda kv: (-kv[1], kv[0])),
        "sentiments": sentiments,
        "entities": top_entities,
    }


def _safe(text: str) -> str:
    """标题里的 [] 会被 Markdown 当成链接语法，替换成中文括号更稳。"""
    return str(text or "").replace("[", "【").replace("]", "】").strip()


def snippet(text: str, tokens: list, width: int = 120) -> str:
    """截取包含命中词的摘要片段，比固定截前 N 字更贴近用户的问题。"""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    pos = -1
    for tok in sorted(tokens, key=len, reverse=True):
        found = text.find(tok)
        if found >= 0:
            pos = found
            break
    if pos < 0:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _describe_stats(stats: dict) -> str:
    parts = []
    if stats.get("date_from"):
        span = stats["date_from"]
        if stats.get("date_to") and stats["date_to"] != stats["date_from"]:
            span = f"{stats['date_from']} ~ {stats['date_to']}"
        parts.append(f"时间跨度 {span}")
    if stats.get("sources"):
        parts.append("来源 " + " / ".join(f"{s} {n} 条" for s, n in stats["sources"][:3]))
    if stats.get("sentiments"):
        order = ("正面", "中性", "负面")
        desc = " / ".join(f"{k} {stats['sentiments'][k]}" for k in order if k in stats["sentiments"])
        if desc:
            parts.append("情感 " + desc)
    if stats.get("entities"):
        parts.append("高频实体 " + "、".join(e for e, _ in stats["entities"][:3]))
    return "；".join(parts)


def build_extractive_answer(question: str, hits: list, stats: dict, mode_note: str = "") -> str:
    """无 LLM 时的抽取式回答：命中概览 + 逐条要点 + 来源链接。

    目标不是"像大模型那样说话"，而是让没有配 LLM 的部署也能得到一份
    可以直接读的结论，而不是一串裸链接。
    """
    lines = []
    overview = _describe_stats(stats)
    lines.append(
        f"**库内命中 {len(hits)} 条**与「{_safe(question)}」相关的情报"
        + (f"（{overview}）。" if overview else "。")
    )
    lines.append("")
    lines.append("**要点**")
    for i, h in enumerate(hits, 1):
        meta = " · ".join(x for x in (h.source, h.date, h.sentiment) if x)
        lines.append(f"{i}. **{_safe(h.title)}**" + (f"（{meta}）" if meta else ""))
        text = snippet(h.summary, h.matched)
        if text:
            lines.append(f"   {text}")
        if h.url:
            lines.append(f"   [原文]({h.url})")
        lines.append("")
    tail = "> 当前为**抽取式回答**（未配置 LLM）：以上内容全部来自库内原文，未做生成式改写。"
    if mode_note:
        tail += f"\n>\n> 检索方式：{mode_note}"
    lines.append(tail)
    return "\n".join(lines).strip()


def build_no_hit_answer(question: str, stats: dict, suggestions: list, mode_note: str = "") -> str:
    """零命中时给出可操作的下一步，而不是一句"未检索到"。"""
    lines = [
        f"**未检索到与「{_safe(question)}」直接相关的情报。**",
        "",
        f"已检索库内 {stats.get('scanned', 0)} 篇情报（不限时间范围），没有命中匹配词。",
    ]
    if suggestions:
        lines.append("")
        lines.append("**库内当前高频话题**（可以直接点这些词提问）：")
        lines.append("　".join(f"`{word}` {n}" for word, n in suggestions[:8]))

    if suggestions:
        examples = " 或 ".join(f"`{w}`" for w, _ in suggestions[:2])
        rewrite_tip = f"少给限定词，只留核心词：把整句换成 {examples} 这类库内出现过的词；"
    else:
        rewrite_tip = "少给限定词，只留一两个核心词再问一次；"

    lines += [
        "",
        "**可以试试**",
        f"1. {rewrite_tip}",
        "2. 换个说法：同一个概念在本库里可能写作另一种表述（如「双11」与「双十一」）；",
        "3. 点侧边栏「立即采集」抓一批最新资讯后再问。",
    ]
    if mode_note:
        lines += ["", f"> 检索方式：{mode_note}"]
    return "\n".join(lines).strip()


def suggest_topics(conn, limit: int = 8) -> list:
    """从已分析情报的 tags / entities 里统计高频话题，用于零命中时的引导。"""
    counter = {}
    try:
        rows = conn.execute("SELECT tags, entities FROM analysis").fetchall()
    except Exception:
        return []
    for row in rows:
        for field_name in ("tags", "entities"):
            try:
                values = row[field_name]
            except (TypeError, IndexError):
                values = None
            items = json.loads(values) if values else []
            for item in items:
                word = str(item).strip()
                if len(word) < 2 or len(word) > 12:
                    continue
                counter[word] = counter.get(word, 0) + 1
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]


def build_llm_messages(question: str, hits: list, history: list = None) -> list:
    """组装带编号来源的对话消息，要求模型用 [1][2] 标注引用。"""
    context = []
    for i, h in enumerate(hits, 1):
        meta = " · ".join(x for x in (h.source, h.date) if x)
        summary = re.sub(r"\s+", " ", h.summary or "")[:220]
        context.append(f"[{i}] 《{h.title}》{f'（{meta}）' if meta else ''}：{summary}")

    system = (
        "你是行业情报分析助手。只能依据给定的『检索到的情报』回答问题，"
        "不得引入资料之外的数字、结论或来源。回答要求："
        "先给出直接结论，再分点列出支撑事实；"
        "每个事实后用 [编号] 标注它来自哪一条情报（例如 [1][3]）；"
        "如果资料不足以回答，就明确说明缺口，并指出还需要哪些数据。"
    )
    messages = [{"role": "system", "content": system}]
    for msg in (history or [])[-4:]:
        role = "assistant" if msg.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": str(msg.get("content", ""))[:600]})
    messages.append(
        {
            "role": "user",
            "content": "检索到的情报：\n" + "\n".join(context) + f"\n\n用户问题：{question}",
        }
    )
    return messages


def citation_list(hits: list) -> str:
    """与 [1][2] 编号一一对应的来源列表。"""
    lines = []
    for i, h in enumerate(hits, 1):
        meta = " · ".join(x for x in (h.source, h.date) if x)
        title = _safe(h.title)
        link = f"[{title}]({h.url})" if h.url else title
        lines.append(f"{i}. {link}" + (f" — {meta}" if meta else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def answer_question(
    conn,
    question: str,
    *,
    emb_cfg=None,
    llm_cfg=None,
    llm_client=None,
    top_k: int = 6,
    days: int = None,
    history: list = None,
    candidates: list = None,
    budget: "CallBudget" = None,
) -> QAAnswer:
    """检索 + 组装回答。UI 侧只需把返回的 QAAnswer 渲染出来。

    - ``candidates`` 允许调用方复用已取好的候选集（避免重复查库）；
    - 向量检索失败（未配 key / 维度不符 / 网络异常）自动降级为关键词检索，
      并通过 ``warning`` 告知用户，不再整段抛错；
    - 传入 ``budget`` 时，生成式回答受调用额度约束，超限自动退回抽取式回答。
    """
    question = (question or "").strip()
    tokens = expand_tokens(tokenize(question))
    rows = candidates if candidates is not None else load_candidates(conn, days=days)
    scanned = len(rows)

    if not tokens:
        return QAAnswer(
            answer=(
                "**问题里没有可用于检索的关键词。**\n\n"
                "可以这样问：「拼多多最近有什么动态？」「AI 手机的方向有哪些变化？」"
            ),
            mode="none",
            mode_note="未提取到检索词",
            stats={"hits": 0, "scanned": scanned},
        )

    kw_hits = keyword_rank(rows, tokens, limit=top_k * 3)
    warning = ""
    vec_hits = []
    if emb_cfg is not None and getattr(emb_cfg, "available", False):
        try:
            from openai import OpenAI

            client = OpenAI(api_key=emb_cfg.api_key, base_url=emb_cfg.base_url)
            qvec = np.array(
                client.embeddings.create(model=emb_cfg.model, input=[question]).data[0].embedding,
                dtype=np.float32,
            )
            vec_hits = vector_rank(rows, qvec, limit=top_k * 3)
            if not vec_hits:
                warning = "库内暂无可比向量（文章未向量化或向量模型已更换），本次使用关键词检索。"
        except Exception as exc:  # noqa: BLE001 - 外部依赖不可控，统一降级
            warning = f"向量检索不可用（{type(exc).__name__}），已降级为关键词检索。"

    if vec_hits:
        hits = hybrid_merge(kw_hits, vec_hits, limit=top_k)
        mode = "hybrid"
        mode_note = "向量语义 + 关键词字面混合排序（全库检索，不受看板时间范围限制）"
    else:
        hits = kw_hits[:top_k]
        mode = "keyword"
        mode_note = "关键词加权匹配（标题×3 / 标签·实体×2 / 摘要×1.2，全库检索）"

    stats = hit_stats(hits, scanned=scanned)

    if not hits:
        suggestions = suggest_topics(conn)
        return QAAnswer(
            answer=build_no_hit_answer(question, stats, suggestions, mode_note),
            hits=[],
            mode="none",
            mode_note=mode_note,
            stats=stats,
            warning=warning,
            suggestions=suggestions,
        )

    if llm_cfg is not None and getattr(llm_cfg, "available", False) and llm_client is not None:
        if budget is not None and not budget.allow():
            warning = (warning + " " if warning else "") + (
                f"生成式回答已达频率上限（每小时 {budget.max_per_window} 次），本次改为抽取式回答。"
            )
        else:
            try:
                resp = llm_client.chat.completions.create(
                    model=llm_cfg.model,
                    messages=build_llm_messages(question, hits, history),
                    temperature=0.2,
                    timeout=60,
                )
                body = (resp.choices[0].message.content or "").strip()
                summary_line = f"\n\n---\n**命中 {len(hits)} 条**"
                if stats.get("date_from"):
                    summary_line += f"（{stats['date_from']} ~ {stats['date_to']}）"
                answer = f"{body}{summary_line}\n\n**引用来源**\n{citation_list(hits)}"
                return QAAnswer(
                    answer=answer, hits=hits, mode=mode, mode_note=mode_note,
                    stats=stats, warning=warning, llm_used=True,
                )
            except Exception as exc:  # noqa: BLE001
                # 详细原因只进服务端日志：页面面向公网访客，不该暴露供应商与密钥状态
                print(f"[qa] 大模型生成失败：{type(exc).__name__}: {exc}")
                warning = (warning + " " if warning else "") + (
                    f"大模型生成失败（{type(exc).__name__}），已改为抽取式回答。"
                )

    return QAAnswer(
        answer=build_extractive_answer(question, hits, stats, mode_note),
        hits=hits, mode=mode, mode_note=mode_note, stats=stats, warning=warning,
    )
