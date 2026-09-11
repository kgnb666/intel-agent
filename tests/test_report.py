"""report 模块单元测试：日报汇总与 HTML/纯文本渲染（标题/摘要/情感标签、空数据、转义、降级说明）。"""
import json

import pytest

from src.report import collect_daily, render_html, render_plain_text
from src.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


def _insert(storage, url, title, fetched_at="2026-01-01 08:00:00", source="测试源", summary=""):
    storage.conn.execute(
        """INSERT INTO articles (url, title, summary, source, published, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (url, title, summary, source, "", fetched_at),
    )
    storage.conn.commit()
    return storage.conn.execute(
        "SELECT id FROM articles WHERE url = ?", (url,)
    ).fetchone()["id"]


def _analyze(storage, aid, sentiment="中性", conf=0.5, summary="AI 摘要", tags=None):
    storage.conn.execute(
        """INSERT INTO analysis
           (article_id, summary_ai, sentiment, sentiment_conf, tags, entities, analyzed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (aid, summary, sentiment, conf, json.dumps(tags or ["电商"], ensure_ascii=False),
         "[]", "2026-01-01 09:00:00"),
    )
    storage.conn.execute("UPDATE articles SET analyzed = 1 WHERE id = ?", (aid,))
    storage.conn.commit()


class TestCollectDaily:
    def test_empty_database(self, storage):
        data = collect_daily(storage, day="2026-01-01")
        assert data["total"] == 0
        assert data["analyzed"] == 0
        assert data["sentiment_dist"] == {"正面": 0, "中性": 0, "负面": 0}
        assert data["top_events"] == []
        assert data["trends"] == []

    def test_filters_by_day(self, storage):
        _insert(storage, "https://e.com/a", "当日", fetched_at="2026-01-01 08:00:00")
        _insert(storage, "https://e.com/b", "其他日", fetched_at="2026-01-02 08:00:00")
        data = collect_daily(storage, day="2026-01-01")
        assert data["total"] == 1
        assert data["all"][0]["title"] == "当日"

    def test_sentiment_distribution(self, storage):
        for i, (senti, conf) in enumerate([("正面", 0.9), ("中性", 0.5), ("负面", 0.8)]):
            aid = _insert(storage, f"https://e.com/{i}", f"标题{i}")
            _analyze(storage, aid, sentiment=senti, conf=conf)
        data = collect_daily(storage, day="2026-01-01")
        assert data["analyzed"] == 3
        assert data["sentiment_dist"] == {"正面": 1, "中性": 1, "负面": 1}

    def test_importance_ranks_high_conf_non_neutral_first(self, storage):
        cases = [("中性", 0.9), ("负面", 0.8), ("正面", 0.5)]
        ids = []
        for i, (senti, conf) in enumerate(cases):
            aid = _insert(storage, f"https://e.com/{i}", f"标题{i}")
            _analyze(storage, aid, sentiment=senti, conf=conf)
            ids.append(aid)
        top = collect_daily(storage, day="2026-01-01")["top_events"]
        assert [t["id"] for t in top] == [ids[1], ids[2], ids[0]]

    def test_trends_included(self, storage):
        storage.save_trend("2026-01-01", [], "趋势内容")
        data = collect_daily(storage, day="2026-01-01")
        assert data["trends"][0]["content"] == "趋势内容"


class TestRenderHtml:
    def _data(self):
        return {
            "day": "2026-01-01",
            "total": 1,
            "analyzed": 1,
            "sentiment_dist": {"正面": 0, "中性": 0, "负面": 1},
            "top_events": [{
                "id": 1,
                "title": "拼多多财报",
                "url": "https://e.com/1",
                "source": "测试源",
                "summary_ai": "营收超预期",
                "sentiment": "负面",
                "sentiment_conf": 0.9,
                "tags": "[\"电商\"]",
            }],
            "trends": [],
            "all": [],
        }

    def test_contains_title_summary_sentiment_and_link(self):
        html = render_html(self._data(), "电商与消费")
        assert "电商与消费" in html
        assert "拼多多财报" in html
        assert "营收超预期" in html
        assert "负面" in html
        assert 'href="https://e.com/1"' in html

    def test_empty_data_still_renders(self):
        data = {
            "day": "2026-01-01",
            "total": 0,
            "analyzed": 0,
            "sentiment_dist": {"正面": 0, "中性": 0, "负面": 0},
            "top_events": [],
            "trends": [],
            "all": [],
        }
        html = render_html(data, "电商与消费")
        assert "当日无已分析事件" in html
        assert "新增 0 条" in html
        assert "正面 0 · 中性 0 · 负面 0" in html

    def test_escapes_injected_html(self):
        data = self._data()
        data["top_events"][0]["title"] = "<script>alert(1)</script>"
        html = render_html(data, "电商与消费")
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_degraded_notes_rendered(self):
        html = render_html(
            self._data(), "电商与消费",
            degraded_notes=["采集环节失败：boom", "未配置 LLM API key"],
        )
        assert "本次流水线部分环节降级" in html
        assert "采集环节失败：boom" in html
        assert "未配置 LLM API key" in html

    def test_no_degraded_block_without_notes(self):
        html = render_html(self._data(), "电商与消费")
        assert "部分环节降级" not in html

    def test_trend_block_rendered(self):
        data = self._data()
        data["trends"] = [{"week_start": "2026-01-01", "content": "本周趋势脉络内容"}]
        html = render_html(data, "电商与消费")
        assert "📈 本周趋势脉络" in html
        assert "本周趋势脉络内容" in html


class TestRenderPlainText:
    def _data(self):
        return {
            "day": "2026-01-01",
            "total": 1,
            "analyzed": 1,
            "sentiment_dist": {"正面": 0, "中性": 0, "负面": 1},
            "top_events": [{
                "id": 1,
                "title": "拼多多财报",
                "url": "https://e.com/1",
                "source": "测试源",
                "summary_ai": "营收超预期",
                "sentiment": "负面",
                "sentiment_conf": 0.9,
                "tags": "[\"电商\"]",
            }],
            "trends": [],
            "all": [],
        }

    def test_contains_title_summary_sentiment_and_link(self):
        text = render_plain_text(self._data(), "电商与消费")
        assert "【电商与消费 · 每日情报日报】" in text
        assert "拼多多财报" in text
        assert "营收超预期" in text
        assert "[负面]" in text
        assert "https://e.com/1" in text

    def test_empty_data_still_renders(self):
        data = {
            "day": "2026-01-01",
            "total": 0,
            "analyzed": 0,
            "sentiment_dist": {"正面": 0, "中性": 0, "负面": 0},
            "top_events": [],
            "trends": [],
            "all": [],
        }
        text = render_plain_text(data, "电商与消费")
        assert "当日无已分析事件" in text
        assert "新增 0 条" in text
        assert "正面 0 · 中性 0 · 负面 0" in text

    def test_degraded_notes_rendered(self):
        text = render_plain_text(
            self._data(), "电商与消费",
            degraded_notes=["采集环节失败：boom", "未配置 LLM API key"],
        )
        assert "⚠️ 本次流水线部分环节降级" in text
        assert "采集环节失败：boom" in text
        assert "未配置 LLM API key" in text

    def test_trend_block_rendered(self):
        data = self._data()
        data["trends"] = [{"week_start": "2026-01-01", "content": "本周趋势脉络内容"}]
        text = render_plain_text(data, "电商与消费")
        assert "📈 本周趋势脉络" in text
        assert "本周趋势脉络内容" in text

