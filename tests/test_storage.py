"""storage 模块单元测试：建表、URL 去重、待分析筛选、分析事务、embedding 与趋势读写。"""
import json

import pytest

from src.storage import Storage

ARTICLE = {
    "url": "https://example.com/a",
    "title": "拼多多发布 Q2 财报",
    "summary": "营收超预期",
    "source": "测试源",
    "published": "2026-01-01T08:00:00",
}

ANALYSIS_OK = {
    "summary": "拼多多 Q2 营收超预期",
    "sentiment": "正面",
    "confidence": 0.9,
    "tags": ["电商", "财报"],
    "entities": ["拼多多"],
}


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


def _insert(storage, url, title="标题", fetched_at="2026-01-01 08:00:00", **kw):
    storage.conn.execute(
        """INSERT INTO articles (url, title, summary, source, published, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (url, title, kw.get("summary", ""), kw.get("source", "测试源"),
         kw.get("published", ""), fetched_at),
    )
    storage.conn.commit()
    return storage.conn.execute(
        "SELECT id FROM articles WHERE url = ?", (url,)
    ).fetchone()["id"]


class TestSchema:
    def test_tables_created(self, storage):
        tables = {
            r[0]
            for r in storage.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert {"articles", "analysis", "trends"} <= tables

    def test_articles_columns_include_rag_fields(self, storage):
        cols = [r[1] for r in storage.conn.execute("PRAGMA table_info(articles)")]
        assert {"url", "title", "analyzed", "embedding", "related_event"} <= set(cols)

    def test_analysis_and_trends_columns(self, storage):
        analysis_cols = [r[1] for r in storage.conn.execute("PRAGMA table_info(analysis)")]
        assert {"summary_ai", "sentiment", "sentiment_conf", "tags", "tokens_used"} <= set(analysis_cols)
        trends_cols = [r[1] for r in storage.conn.execute("PRAGMA table_info(trends)")]
        assert {"week_start", "cluster", "content", "created_at"} <= set(trends_cols)


class TestInsertDedup:
    def test_insert_new_article_returns_true(self, storage):
        assert storage.insert_article(dict(ARTICLE)) is True
        assert storage.count() == 1

    def test_duplicate_url_skipped(self, storage):
        assert storage.insert_article(dict(ARTICLE)) is True
        assert storage.insert_article(dict(ARTICLE)) is False
        assert storage.count() == 1

    def test_same_title_different_url_both_kept(self, storage):
        assert storage.insert_article(dict(ARTICLE)) is True
        assert storage.insert_article(dict(ARTICLE, url="https://example.com/b")) is True
        assert storage.count() == 2

    def test_recent_titles_newest_first_with_limit(self, storage):
        for i in range(3):
            storage.insert_article(
                {**ARTICLE, "url": f"https://e.com/{i}", "title": f"标题{i}"}
            )
        assert storage.recent_titles(2) == ["标题2", "标题1"]


class TestUnanalyzed:
    def test_unanalyzed_returns_only_pending_in_order(self, storage):
        ids = [_insert(storage, f"https://e.com/{i}", title=f"标题{i}") for i in range(3)]
        storage.save_analysis(ids[0], dict(ANALYSIS_OK), tokens=10)
        rows = storage.unanalyzed_articles(10)
        assert [r["id"] for r in rows] == ids[1:]

    def test_unanalyzed_respects_limit(self, storage):
        for i in range(3):
            _insert(storage, f"https://e.com/{i}")
        assert len(storage.unanalyzed_articles(2)) == 2


class TestSaveAnalysis:
    def test_writes_analysis_and_marks_article(self, storage):
        aid = _insert(storage, "https://e.com/1")
        storage.save_analysis(aid, dict(ANALYSIS_OK), tokens=42)
        row = storage.conn.execute(
            "SELECT * FROM analysis WHERE article_id = ?", (aid,)
        ).fetchone()
        assert row["summary_ai"] == ANALYSIS_OK["summary"]
        assert row["sentiment"] == "正面"
        assert row["sentiment_conf"] == 0.9
        assert json.loads(row["tags"]) == ["电商", "财报"]
        assert json.loads(row["entities"]) == ["拼多多"]
        assert row["tokens_used"] == 42
        art = storage.conn.execute(
            "SELECT analyzed FROM articles WHERE id = ?", (aid,)
        ).fetchone()
        assert art["analyzed"] == 1
        assert storage.unanalyzed_articles(10) == []

    def test_rollback_keeps_consistency_on_error(self, storage):
        aid = _insert(storage, "https://e.com/1")
        bad = dict(ANALYSIS_OK, tags=[object()])  # json.dumps 必然失败
        with pytest.raises(TypeError):
            storage.save_analysis(aid, bad, tokens=1)
        n = storage.conn.execute("SELECT COUNT(*) AS n FROM analysis").fetchone()["n"]
        assert n == 0
        art = storage.conn.execute(
            "SELECT analyzed FROM articles WHERE id = ?", (aid,)
        ).fetchone()
        assert art["analyzed"] == 0  # 事务回滚，两张表状态一致


class TestEmbedding:
    def test_save_and_read_roundtrip(self, storage):
        aid = _insert(storage, "https://e.com/1")
        blob = b"\x00\x00\x80?\x00\x00\x00@"
        storage.save_embedding(aid, blob)
        rows = storage.embedded_articles()
        assert len(rows) == 1
        assert rows[0]["id"] == aid
        assert rows[0]["embedding"] == blob

    def test_without_embedding_excludes_embedded(self, storage):
        aid = _insert(storage, "https://e.com/1")
        storage.save_embedding(aid, b"x")
        assert storage.articles_without_embedding() == []

    def test_without_embedding_prefers_ai_summary(self, storage):
        aid = _insert(storage, "https://e.com/1", summary="原文摘要")
        storage.save_analysis(aid, dict(ANALYSIS_OK), tokens=5)
        rows = storage.articles_without_embedding()
        assert rows[0]["id"] == aid
        assert rows[0]["text"] == ANALYSIS_OK["summary"]

    def test_embedded_articles_before_id_and_limit(self, storage):
        ids = [_insert(storage, f"https://e.com/{i}") for i in range(3)]
        for i in ids:
            storage.save_embedding(i, b"v")
        rows = storage.embedded_articles(before_id=ids[2])
        assert [r["id"] for r in rows] == [ids[1], ids[0]]
        rows2 = storage.embedded_articles(limit=1)
        assert len(rows2) == 1 and rows2[0]["id"] == ids[2]


class TestRelatedTrends:
    def test_save_related_roundtrip(self, storage):
        aid = _insert(storage, "https://e.com/1")
        related = [{"id": 2, "title": "历史", "similarity": 0.8}]
        storage.save_related(aid, related)
        raw = storage.conn.execute(
            "SELECT related_event FROM articles WHERE id = ?", (aid,)
        ).fetchone()["related_event"]
        assert json.loads(raw) == related

    def test_save_trend_and_latest_order(self, storage):
        storage.save_trend("2026-08-24", [{"id": 1}], "趋势一")
        storage.save_trend("2026-08-31", [{"id": 2}], "趋势二")
        rows = storage.latest_trends()
        assert [r["content"] for r in rows] == ["趋势二", "趋势一"]
        assert json.loads(rows[0]["cluster"]) == [{"id": 2}]

    def test_latest_trends_limit(self, storage):
        for i in range(3):
            storage.save_trend("2026-08-24", [], f"t{i}")
        assert len(storage.latest_trends(2)) == 2


class TestSentReports:
    def test_has_report_sent_initially_false(self, storage):
        assert storage.has_report_sent("2026-09-09") is False

    def test_record_report_sent_marks_true(self, storage):
        storage.record_report_sent("2026-09-09")
        assert storage.has_report_sent("2026-09-09") is True
        assert storage.has_report_sent("2026-09-10") is False

    def test_record_report_sent_idempotent(self, storage):
        storage.record_report_sent("2026-09-09")
        storage.record_report_sent("2026-09-09")
        assert storage.has_report_sent("2026-09-09") is True
        count = storage.conn.execute(
            "SELECT COUNT(*) as n FROM sent_reports WHERE day = '2026-09-09'"
        ).fetchone()["n"]
        assert count == 1


class TestSentAlerts:
    def test_is_alert_sent_initially_false(self, storage):
        assert storage.is_alert_sent(1, 101) is False

    def test_record_alert_sent_marks_true(self, storage):
        storage.record_alert_sent(1, 101)
        assert storage.is_alert_sent(1, 101) is True
        assert storage.is_alert_sent(1, 102) is False
        assert storage.is_alert_sent(2, 101) is False

    def test_record_alert_sent_idempotent(self, storage):
        storage.record_alert_sent(1, 101)
        storage.record_alert_sent(1, 101)
        assert storage.is_alert_sent(1, 101) is True
        count = storage.conn.execute(
            "SELECT COUNT(*) as n FROM sent_alerts WHERE event_id = 1 AND article_id = 101"
        ).fetchone()["n"]
        assert count == 1


class TestInsertArticles:
    def test_batch_insert_empty_returns_zero(self, storage):
        assert storage.insert_articles([]) == 0

    def test_batch_insert_skips_duplicate_urls(self, storage):
        items = [
            {"url": "https://a.com/1", "title": "标题1", "source": "源A"},
            {"url": "https://a.com/2", "title": "标题2", "source": "源A"},
            {"url": "https://a.com/1", "title": "标题1重复", "source": "源A"},
        ]
        inserted = storage.insert_articles(items)
        assert inserted == 2
        assert storage.count() == 2


class TestPruneHistory:
    def test_prune_history_zero_days_skips(self, storage):
        res = storage.prune_history(0)
        assert res == {"deleted_articles": 0, "deleted_analysis": 0}

    def test_prune_history_deletes_old_and_cascades(self, storage):
        # 1. 插入老文章 (2025年)
        storage.conn.execute(
            """INSERT INTO articles (url, title, source, fetched_at)
               VALUES ('https://old.com', '老文章', '源A', '2025-01-01 12:00:00')"""
        )
        old_id = storage.conn.execute("SELECT id FROM articles WHERE url = 'https://old.com'").fetchone()["id"]
        storage.save_analysis(old_id, {"summary": "老摘要", "sentiment": "中性", "confidence": 0.5, "tags": [], "entities": []}, tokens=10)
        storage.record_alert_sent(99, old_id)

        # 2. 插入新文章
        storage.insert_article({"url": "https://new.com", "title": "新文章", "source": "源B"})
        new_id = storage.conn.execute("SELECT id FROM articles WHERE url = 'https://new.com'").fetchone()["id"]
        storage.save_analysis(new_id, {"summary": "新摘要", "sentiment": "正面", "confidence": 0.9, "tags": [], "entities": []}, tokens=20)

        # 3. 修剪 90 天前数据
        res = storage.prune_history(retention_days=90)
        assert res["deleted_articles"] == 1
        assert res["deleted_analysis"] == 1

        # 验证老文章级联删除
        assert storage.conn.execute("SELECT 1 FROM articles WHERE id = ?", (old_id,)).fetchone() is None
        assert storage.conn.execute("SELECT 1 FROM analysis WHERE article_id = ?", (old_id,)).fetchone() is None
        assert storage.is_alert_sent(99, old_id) is False

        # 验证新文章完好保留
        assert storage.conn.execute("SELECT 1 FROM articles WHERE id = ?", (new_id,)).fetchone() is not None
        assert storage.conn.execute("SELECT 1 FROM analysis WHERE article_id = ?", (new_id,)).fetchone() is not None


class TestFeedHealth:
    def test_feed_health_initial_not_broken(self, storage):
        assert storage.is_feed_broken("https://rss.example.com") is False

    def test_consecutive_failures_triggers_circuit_breaker(self, storage):
        url = "https://broken.example.com/rss"
        for i in range(4):
            storage.record_feed_result(url, "故障源", success=False, error=f"500 error {i}")
            assert storage.is_feed_broken(url, max_failures=5) is False

        # 第 5 次失败，触发熔断
        storage.record_feed_result(url, "故障源", success=False, error="500 error 5")
        assert storage.is_feed_broken(url, max_failures=5) is True

        # 成功一次后，熔断状态重置
        storage.record_feed_result(url, "故障源", success=True)
        assert storage.is_feed_broken(url, max_failures=5) is False



