"""存储模块：SQLite 持久化，建表、插入、去重查询。"""
import json
import os
import sqlite3
from datetime import datetime, timedelta

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    url         TEXT UNIQUE NOT NULL,          -- 以 URL 作为天然去重键
    title       TEXT NOT NULL,
    summary     TEXT,
    source      TEXT NOT NULL,
    published   TEXT,                          -- 源站发布时间（ISO 字符串）
    fetched_at  TEXT NOT NULL,                 -- 入库时间
    analyzed    INTEGER DEFAULT 0              -- 是否已完成 LLM 分析（后续阶段用）
);
CREATE INDEX IF NOT EXISTS idx_articles_fetched ON articles(fetched_at);
CREATE INDEX IF NOT EXISTS idx_articles_analyzed ON articles(analyzed);

CREATE TABLE IF NOT EXISTS analysis (
    article_id INTEGER PRIMARY KEY REFERENCES articles(id),
    summary_ai    TEXT,                -- LLM 生成摘要（80 字内）
    sentiment     TEXT,                -- 正面 / 中性 / 负面
    sentiment_conf REAL,               -- 情感置信度 0~1
    tags          TEXT,                -- 行业标签，JSON 数组字符串
    entities      TEXT,                -- 关键实体（公司/产品），JSON 数组字符串
    tokens_used   INTEGER DEFAULT 0,   -- 本条分析消耗的 token，用于成本核算
    analyzed_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trends (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start  TEXT NOT NULL,           -- 趋势所属周的周一日期
    cluster     TEXT,                    -- 该趋势包含的事件（JSON 数组：article id/title）
    content     TEXT NOT NULL,           -- LLM 生成的趋势脉络段落
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sent_reports (
    day         TEXT PRIMARY KEY,        -- 报告日期 YYYY-MM-DD
    sent_at     TEXT NOT NULL            -- 发送完成时间
);

CREATE TABLE IF NOT EXISTS sent_alerts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    INTEGER NOT NULL,
    article_id  INTEGER NOT NULL,
    sent_at     TEXT NOT NULL,
    UNIQUE(event_id, article_id)
);

CREATE TABLE IF NOT EXISTS feed_health (
    url                  TEXT PRIMARY KEY,
    source_name          TEXT NOT NULL,
    consecutive_failures INTEGER DEFAULT 0,
    last_error           TEXT,
    last_success_at      TEXT,
    updated_at           TEXT NOT NULL
);
"""


def _ensure_column(conn, table: str, column: str, ddl: str):
    """轻量迁移：老库缺列时补上，避免重建数据库丢数据。"""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


class Storage:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, timeout=10.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=10000;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.executescript(SCHEMA)
        # articles 表追加 RAG 阶段字段：向量与关联事件链
        _ensure_column(self.conn, "articles", "embedding", "embedding BLOB")
        _ensure_column(self.conn, "articles", "related_event", "related_event TEXT")
        self.conn.commit()

    def insert_article(self, item: dict) -> bool:
        """插入一条情报，URL 已存在则跳过。返回是否为新插入。"""
        try:
            self.conn.execute(
                """INSERT INTO articles (url, title, summary, source, published, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    item["url"],
                    item["title"],
                    item.get("summary", ""),
                    item["source"],
                    item.get("published", ""),
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def insert_articles(self, items: list) -> int:
        """批量插入情报，URL 已存在则跳过。单事务提交，返回实际新插入篇数。"""
        if not items:
            return 0
        inserted = 0
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            for item in items:
                try:
                    self.conn.execute(
                        """INSERT INTO articles (url, title, summary, source, published, fetched_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            item["url"],
                            item["title"],
                            item.get("summary", ""),
                            item["source"],
                            item.get("published", ""),
                            now_iso,
                        ),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    continue
        return inserted


    def count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
        return row["n"]

    def recent_titles(self, limit: int = 500) -> list:
        rows = self.conn.execute(
            "SELECT title FROM articles ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [r["title"] for r in rows]

    def close(self):
        self.conn.close()

    # ---------- LLM 分析阶段 ----------

    def unanalyzed_articles(self, limit: int) -> list:
        """取待分析文章（按入库顺序，先进先出）。"""
        rows = self.conn.execute(
            "SELECT id, title, summary, source, published FROM articles "
            "WHERE analyzed = 0 ORDER BY id LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_analysis(self, article_id: int, result: dict, tokens: int):
        """写入分析结果并置 analyzed 标记。用事务保证两张表状态一致。"""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO analysis
                   (article_id, summary_ai, sentiment, sentiment_conf, tags, entities, tokens_used, analyzed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    article_id,
                    result["summary"],
                    result["sentiment"],
                    result["confidence"],
                    json.dumps(result["tags"], ensure_ascii=False),
                    json.dumps(result["entities"], ensure_ascii=False),
                    tokens,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            self.conn.execute(
                "UPDATE articles SET analyzed = 1 WHERE id = ?", (article_id,)
            )

    # ---------- RAG 阶段 ----------

    def articles_without_embedding(self, limit: int = 200) -> list:
        """取已向量化缺失的文章。用分析摘要（没有则原文摘要）作为向量文本。"""
        rows = self.conn.execute(
            """SELECT a.id, a.title, a.published, a.fetched_at,
                      COALESCE(an.summary_ai, a.summary) AS text
               FROM articles a
               LEFT JOIN analysis an ON an.article_id = a.id
               WHERE a.embedding IS NULL
               ORDER BY a.id LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def save_embedding(self, article_id: int, vector: bytes):
        with self.conn:
            self.conn.execute(
                "UPDATE articles SET embedding = ? WHERE id = ?", (vector, article_id)
            )

    def embedded_articles(self, before_id: int = None, limit: int = 1000) -> list:
        """取已有向量的文章，用于相似度检索。before_id 限定"历史"范围。"""
        sql = ("SELECT id, title, embedding, fetched_at FROM articles "
               "WHERE embedding IS NOT NULL")
        params = []
        if before_id is not None:
            sql += " AND id < ?"
            params.append(before_id)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def save_related(self, article_id: int, related: list):
        with self.conn:
            self.conn.execute(
                "UPDATE articles SET related_event = ? WHERE id = ?",
                (json.dumps(related, ensure_ascii=False), article_id),
            )

    def save_trend(self, week_start: str, cluster: list, content: str):
        with self.conn:
            self.conn.execute(
                "INSERT INTO trends (week_start, cluster, content, created_at) VALUES (?, ?, ?, ?)",
                (
                    week_start,
                    json.dumps(cluster, ensure_ascii=False),
                    content,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )

    def latest_trends(self, limit: int = 5) -> list:
        rows = self.conn.execute(
            "SELECT * FROM trends ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---------- 日报发送记录 ----------

    def has_report_sent(self, day: str) -> bool:
        """检查指定日期的日报是否已发送过。"""
        row = self.conn.execute(
            "SELECT 1 FROM sent_reports WHERE day = ?", (day,)
        ).fetchone()
        return row is not None

    def record_report_sent(self, day: str):
        """记录指定日期的日报发送完成状态。"""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sent_reports (day, sent_at) VALUES (?, ?)",
                (day, datetime.now().isoformat(timespec="seconds")),
            )

    # ---------- 告警发送记录 ----------

    def is_alert_sent(self, event_id: int, article_id: int) -> bool:
        """检查指定事件的核心报道是否已发送过告警。"""
        row = self.conn.execute(
            "SELECT 1 FROM sent_alerts WHERE event_id = ? AND article_id = ?",
            (event_id, article_id),
        ).fetchone()
        return row is not None

    def record_alert_sent(self, event_id: int, article_id: int):
        """记录指定事件的核心报道告警发送完成状态。"""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sent_alerts (event_id, article_id, sent_at) VALUES (?, ?, ?)",
                (event_id, article_id, datetime.now().isoformat(timespec="seconds")),
            )

    # ---------- 数据生命周期修剪 (TTL) ----------

    def prune_history(self, retention_days: int = 90) -> dict:
        """删除超过 retention_days 天的历史文章，级联清理 analysis 和 sent_alerts。
        完成后触发 WAL checkpoint 释放空间。
        """
        if retention_days <= 0:
            return {"deleted_articles": 0, "deleted_analysis": 0}
        cutoff = (datetime.now() - timedelta(days=retention_days)).date().isoformat()
        with self.conn:
            cur_an = self.conn.execute(
                "DELETE FROM analysis WHERE article_id IN (SELECT id FROM articles WHERE date(fetched_at) < ?)",
                (cutoff,),
            )
            deleted_an = cur_an.rowcount
            self.conn.execute(
                "DELETE FROM sent_alerts WHERE article_id IN (SELECT id FROM articles WHERE date(fetched_at) < ?)",
                (cutoff,),
            )
            cur_art = self.conn.execute(
                "DELETE FROM articles WHERE date(fetched_at) < ?", (cutoff,)
            )
            deleted_art = cur_art.rowcount

        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass

        return {"deleted_articles": deleted_art, "deleted_analysis": deleted_an}

    # ---------- 数据源健康度管理 ----------

    def record_feed_result(self, url: str, name: str, success: bool, error: str = None):
        """记录数据源抓取结果并更新连续失败计数。"""
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.conn:
            row = self.conn.execute(
                "SELECT consecutive_failures FROM feed_health WHERE url = ?", (url,)
            ).fetchone()
            if success:
                if row:
                    self.conn.execute(
                        """UPDATE feed_health SET consecutive_failures = 0, last_error = NULL,
                                  last_success_at = ?, updated_at = ? WHERE url = ?""",
                        (now_iso, now_iso, url),
                    )
                else:
                    self.conn.execute(
                        """INSERT INTO feed_health (url, source_name, consecutive_failures, last_error, last_success_at, updated_at)
                           VALUES (?, ?, 0, NULL, ?, ?)""",
                        (url, name, now_iso, now_iso),
                    )
            else:
                prev_failures = row["consecutive_failures"] if row else 0
                new_failures = prev_failures + 1
                if row:
                    self.conn.execute(
                        """UPDATE feed_health SET consecutive_failures = ?, last_error = ?,
                                  updated_at = ? WHERE url = ?""",
                        (new_failures, str(error or "")[:500], now_iso, url),
                    )
                else:
                    self.conn.execute(
                        """INSERT INTO feed_health (url, source_name, consecutive_failures, last_error, last_success_at, updated_at)
                           VALUES (?, ?, ?, ?, NULL, ?)""",
                        (url, name, new_failures, str(error or "")[:500], now_iso),
                    )

    def is_feed_broken(self, url: str, max_failures: int = 5) -> bool:
        """检查数据源是否已连续失败达到阈值处于熔断状态。"""
        row = self.conn.execute(
            "SELECT consecutive_failures FROM feed_health WHERE url = ?", (url,)
        ).fetchone()
        if row and (row["consecutive_failures"] or 0) >= max_failures:
            return True
        return False


