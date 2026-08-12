"""采集模块：RSS 抓取 + 关键词过滤 + 清洗去重。"""
import re
import time
from difflib import SequenceMatcher

import feedparser
import requests

_UA = {"User-Agent": "intel-agent/0.1 (+https://github.com/)"}
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")


def _clean_text(html: str) -> str:
    """去除 HTML 标签、压缩空白。"""
    text = _TAG_RE.sub(" ", html or "")
    return _SPACE_RE.sub(" ", text).strip()


def _match_keywords(text: str, keywords: list) -> bool:
    return any(k in text for k in keywords)


def _is_similar(title: str, existing_titles: list, threshold: float) -> bool:
    """标题相似度去重：同一事件不同来源的标题往往高度相似。"""
    for t in existing_titles:
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
    """执行一轮完整采集：抓取 -> 关键词过滤 -> 相似度去重 -> 入库。"""
    keywords = cfg["industry"]["keywords"]
    max_items = cfg["crawl"]["max_items_per_source"]
    threshold = cfg["crawl"]["title_similarity_threshold"]

    stats = {"fetched": 0, "matched": 0, "inserted": 0, "errors": []}
    existing_titles = storage.recent_titles()

    for source in cfg["sources"]:
        try:
            items = fetch_rss(source, max_items)
        except Exception as e:  # 单源失败不阻断整体任务
            stats["errors"].append(f"{source['name']}: {e}")
            continue

        stats["fetched"] += len(items)
        for item in items:
            text = item["title"] + " " + item["summary"]
            if not _match_keywords(text, keywords):
                continue
            stats["matched"] += 1
            if _is_similar(item["title"], existing_titles, threshold):
                continue
            if storage.insert_article(item):
                stats["inserted"] += 1
                existing_titles.append(item["title"])

    return stats
