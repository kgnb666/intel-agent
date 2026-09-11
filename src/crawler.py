"""采集模块：RSS 抓取 + 关键词过滤 + 清洗去重。"""
import html
import re
import time
from difflib import SequenceMatcher

import feedparser
import requests

_UA = {"User-Agent": "intel-agent/0.1 (+https://github.com/)"}
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean_text(html_text: str) -> str:
    """去除 HTML 标签、反转义 HTML 实体并压缩空白。"""
    text = _TAG_RE.sub(" ", html_text or "")
    text = html.unescape(text)
    return _SPACE_RE.sub(" ", text).strip()


def _match_keywords(text: str, keywords: list) -> bool:
    """关键词过滤：统一小写后匹配，支持英文品牌与缩写大小写不敏感匹配。"""
    text_lower = (text or "").lower()
    return any(k.lower() in text_lower for k in keywords)


def _is_similar(title: str, existing_titles: list, threshold: float) -> bool:
    """标题相似度去重：引入长度剪枝与字符集交集快速初筛 + SequenceMatcher 精确计算。"""
    if not existing_titles or not title:
        return False

    title_len = len(title)
    title_chars = set(title)

    for t in existing_titles:
        t_len = len(t)
        if not t_len:
            continue
        # 1. 长度绝对上限快速剪枝（数学必然：ratio <= 2 * min / (len1 + len2)）
        if 2.0 * min(title_len, t_len) / (title_len + t_len) < threshold:
            continue
        # 2. 字符集交集重叠率初筛（Fast reject）
        t_chars = set(t)
        min_chars = min(len(title_chars), len(t_chars))
        if min_chars == 0 or len(title_chars & t_chars) / min_chars < 0.6:
            continue
        # 3. 精确计算
        if SequenceMatcher(None, title, t).ratio() >= threshold:
            return True
    return False


def _get_with_retry(url: str, retries: int = 2, interval: int = 5):
    """带重试的 GET：RSS 源偶发超时/5xx 属正常波动，重试比直接放弃更划算。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=_UA, timeout=25)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                time.sleep(interval)
    raise last_err


def fetch_rss(source: dict, max_items: int) -> list:
    """抓取单个 RSS 源，返回原始条目列表。"""
    resp = _get_with_retry(source["url"])
    resp.raise_for_status()
    feed = feedparser.parse(resp.content)
    items = []
    for entry in feed.entries[:max_items]:
        items.append(
            {
                "url": entry.get("link", ""),
                "title": _clean_text(entry.get("title", "")),
                "summary": _clean_text(entry.get("summary", ""))[:500],
                "source": source["name"],
                "published": entry.get("published", entry.get("updated", "")),
            }
        )
    return [i for i in items if i["url"] and i["title"]]


def run_crawl(cfg: dict, storage) -> dict:
    """执行一轮完整采集：抓取 -> 关键词过滤 -> 相似度去重 -> 批量入库。"""
    keywords = cfg["industry"]["keywords"]
    max_items = cfg["crawl"]["max_items_per_source"]
    threshold = cfg["crawl"]["title_similarity_threshold"]

    stats = {"fetched": 0, "matched": 0, "inserted": 0, "errors": []}
    existing_titles = storage.recent_titles() if hasattr(storage, "recent_titles") else []

    for source in cfg["sources"]:
        url = source.get("url", "")
        name = source.get("name", "")

        # 1. 检查数据源是否处于连续失败熔断状态
        if hasattr(storage, "is_feed_broken") and storage.is_feed_broken(url):
            stats["errors"].append(f"[熔断跳过] {name} 已连续失败多次，暂停抓取")
            continue

        try:
            items = fetch_rss(source, max_items)
            if hasattr(storage, "record_feed_result"):
                storage.record_feed_result(url, name, success=True)
        except Exception as e:  # 单源失败不阻断整体任务
            stats["errors"].append(f"{name}: {e}")
            if hasattr(storage, "record_feed_result"):
                storage.record_feed_result(url, name, success=False, error=str(e))
            continue

        stats["fetched"] += len(items)
        to_insert = []
        for item in items:
            text = item["title"] + " " + item["summary"]
            if not _match_keywords(text, keywords):
                continue
            stats["matched"] += 1
            if _is_similar(item["title"], existing_titles, threshold):
                continue
            to_insert.append(item)
            existing_titles.append(item["title"])

        # 2. 批量单事务入库
        if hasattr(storage, "insert_articles"):
            inserted_count = storage.insert_articles(to_insert)
            stats["inserted"] += inserted_count
        else:
            for item in to_insert:
                if storage.insert_article(item):
                    stats["inserted"] += 1

    return stats

