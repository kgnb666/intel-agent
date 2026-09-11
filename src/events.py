"""事件聚合模块：把 related_event 关联链合并成"事件实体 + 时间线"。

rag 阶段产出的 related_event 是"某篇文章 → 若干历史相似文章"的关联边，
本模块把这些边视为无向图，用并查集求连通分量——每个连通分量就是一个事件实体：
同一事件的多次报道（不同媒体 / 不同时间）被聚合到一起，并按时序排出时间线。

为什么用并查集而不是嵌套循环：关联边可能形成长链（A→B→C）或环，并查集
一次扫描即可求出全部连通分量，复杂度 O(n·α(n))，百级数据量下毫秒级。
"""
import json
import sqlite3
from datetime import datetime, timedelta


def aggregate_events(storage, min_size: int = 2, days: int = None, max_size: int = 15) -> list:
    """把 related_event 链合并成事件实体。

    参数：
        storage: Storage 实例或任意带 Row 工厂的 sqlite3 连接（取 .conn 兼容两者）
        min_size: 事件至少包含的报道数，默认 2（孤立文章不算事件）
        days: 仅聚合最近 N 天（含当天）的文章，None 表示全部历史
        max_size: 单个事件最多保留的报道数，默认 15（防长链传递性漂移与无限膨胀）

    返回按 (成员数降序, 最小文章 id 升序) 排序的事件列表，每个事件：
        id / title / size / min_id / max_id / articles / timeline
    """
    conn = getattr(storage, "conn", storage)
    if conn.row_factory is None:
        conn.row_factory = sqlite3.Row

    since_sql = ""
    params = []
    if days is not None and days > 0:
        since_sql = "AND date(fetched_at) >= ?"
        params.append((datetime.now() - timedelta(days=days - 1)).date().isoformat())

    # 成员池：窗口内全部文章（含被关联但自身无 related_event 的"链尾"文章）
    rows = conn.execute(
        """SELECT a.id, a.title, a.url, a.source, a.published, a.fetched_at,
                  an.sentiment, an.sentiment_conf, an.summary_ai
           FROM articles a LEFT JOIN analysis an ON an.article_id = a.id
           WHERE 1=1 """ + since_sql,
        params,
    ).fetchall()
    if not rows:
        return []

    # 关联边：related_event 非空的行
    edge_rows = conn.execute(
        """SELECT id, related_event FROM articles
           WHERE related_event IS NOT NULL AND related_event != '[]' """ + since_sql,
        params,
    ).fetchall()

    parent = {r["id"]: r["id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in edge_rows:
        for rel in json.loads(r["related_event"]):
            if rel["id"] in parent:
                parent[find(r["id"])] = find(rel["id"])

    groups = {}
    for r in rows:
        groups.setdefault(find(r["id"]), []).append(r)

    events = []
    for members in groups.values():
        if len(members) < min_size:
            continue

        if max_size is not None and max_size > 0 and len(members) > max_size:
            member_ids = {m["id"] for m in members}
            density = {m["id"]: 0.0 for m in members}
            for r in edge_rows:
                src_id = r["id"]
                if src_id in member_ids:
                    try:
                        rels = json.loads(r["related_event"] or "[]")
                    except Exception:
                        rels = []
                    for rel in rels:
                        dst_id = rel.get("id")
                        if dst_id in member_ids:
                            sim = float(rel.get("similarity", 1.0))
                            density[src_id] += sim
                            density[dst_id] += sim

            ranked = sorted(
                members,
                key=lambda m: (density.get(m["id"], 0.0), m["fetched_at"] or "", m["id"]),
                reverse=True,
            )
            members = ranked[:max_size]

        ordered = sorted(members, key=lambda m: (m["fetched_at"] or "", m["id"]))
        event_id = min(m["id"] for m in members)
        articles = [
            {
                "id": m["id"],
                "title": m["title"],
                "url": m["url"],
                "source": m["source"],
                "published": m["published"],
                "fetched_at": m["fetched_at"],
                "sentiment": m["sentiment"],
                "sentiment_conf": m["sentiment_conf"],
                "summary_ai": m["summary_ai"],
            }
            for m in ordered
        ]
        events.append(
            {
                "id": event_id,
                "title": ordered[0]["title"],  # 最早报道的标题作为事件名
                "size": len(members),
                "min_id": min(m["id"] for m in members),
                "max_id": max(m["id"] for m in members),
                "articles": articles,
                "timeline": [
                    {
                        "id": m["id"],
                        "date": (m["fetched_at"] or "")[:10],
                        "title": m["title"],
                        "source": m["source"],
                        "url": m["url"],
                        "sentiment": m["sentiment"],
                    }
                    for m in ordered
                ],
            }
        )

    events.sort(key=lambda e: (-e["size"], e["id"]))
    return events
