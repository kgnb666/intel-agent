"""events 模块单元测试：关联链合并、去重、排序、时间线与窗口过滤。"""
import sqlite3

import pytest

from src.events import aggregate_events
from src.storage import Storage


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


def _insert(storage, url, title, fetched_at="2026-01-01 08:00:00", source="测试源"):
    storage.conn.execute(
        """INSERT INTO articles (url, title, summary, source, published, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (url, title, "", source, "", fetched_at),
    )
    storage.conn.commit()
    return storage.conn.execute(
        "SELECT id FROM articles WHERE url = ?", (url,)
    ).fetchone()["id"]


class TestChainMerge:
    def test_two_article_chain_forms_event(self, storage):
        a = _insert(storage, "https://e.com/a", "事件A报道一")
        b = _insert(storage, "https://e.com/b", "事件A报道二")
        storage.save_related(b, [{"id": a, "title": "事件A报道一", "similarity": 0.9}])
        events = aggregate_events(storage)
        assert len(events) == 1
        ev = events[0]
        assert ev["id"] == a
        assert ev["size"] == 2
        assert [x["id"] for x in ev["timeline"]] == [a, b]

    def test_long_chain_merges_all(self, storage):
        ids = [
            _insert(storage, f"https://e.com/{i}", f"链上报道{i}") for i in range(3)
        ]
        storage.save_related(ids[1], [{"id": ids[0], "title": "t", "similarity": 0.9}])
        storage.save_related(ids[2], [{"id": ids[1], "title": "t", "similarity": 0.9}])
        events = aggregate_events(storage)
        assert len(events) == 1
        assert events[0]["size"] == 3
        assert len(events[0]["articles"]) == 3
        assert len({x["id"] for x in events[0]["articles"]}) == 3  # 链上成员无重复

    def test_diamond_edges_dedup_into_one_event(self, storage):
        ids = [_insert(storage, f"https://e.com/{i}", f"报道{i}") for i in range(3)]
        storage.save_related(ids[1], [{"id": ids[0], "title": "t", "similarity": 0.9}])
        storage.save_related(ids[2], [
            {"id": ids[0], "title": "t", "similarity": 0.9},
            {"id": ids[1], "title": "t", "similarity": 0.8},
        ])
        events = aggregate_events(storage)
        assert len(events) == 1
        assert events[0]["size"] == 3
        assert len(events[0]["articles"]) == 3  # 并查集去重，不会重复收录

    def test_leaf_article_without_own_relations_included(self, storage):
        a = _insert(storage, "https://e.com/a", "链尾报道")
        b = _insert(storage, "https://e.com/b", "新报道")
        storage.save_related(b, [{"id": a, "title": "链尾报道", "similarity": 0.9}])
        events = aggregate_events(storage)
        assert len(events) == 1
        assert {x["id"] for x in events[0]["articles"]} == {a, b}


class TestSeparateAndSort:
    def test_separate_chains_are_distinct_events(self, storage):
        a1 = _insert(storage, "https://e.com/a1", "事件A报道一")
        a2 = _insert(storage, "https://e.com/a2", "事件A报道二")
        storage.save_related(a2, [{"id": a1, "title": "t", "similarity": 0.9}])
        b1 = _insert(storage, "https://e.com/b1", "事件B报道一")
        b2 = _insert(storage, "https://e.com/b2", "事件B报道二")
        storage.save_related(b2, [{"id": b1, "title": "t", "similarity": 0.9}])
        events = aggregate_events(storage)
        assert len(events) == 2
        assert [e["id"] for e in events] == [a1, b1]  # 同规模按最小 id 升序

    def test_sorted_by_size_desc_then_id(self, storage):
        # 事件A：3 篇链（id 1,2,3）；事件B：2 篇链（id 10,11）
        ids_a = [_insert(storage, f"https://e.com/a{i}", f"A{i}") for i in range(3)]
        for i in range(1, 3):
            storage.save_related(ids_a[i], [{"id": ids_a[i - 1], "title": "t", "similarity": 0.9}])
        b1 = _insert(storage, "https://e.com/b1", "B1")
        b2 = _insert(storage, "https://e.com/b2", "B2")
        storage.save_related(b2, [{"id": b1, "title": "t", "similarity": 0.9}])
        events = aggregate_events(storage)
        assert [e["id"] for e in events] == [ids_a[0], b1]
        assert events[0]["size"] == 3 and events[1]["size"] == 2


class TestTimelineAndWindow:
    def test_timeline_chronological(self, storage):
        late = _insert(storage, "https://e.com/late", "较晚报道", fetched_at="2026-01-03 10:00:00")
        early = _insert(storage, "https://e.com/early", "最早报道", fetched_at="2026-01-01 08:00:00")
        mid = _insert(storage, "https://e.com/mid", "中间报道", fetched_at="2026-01-02 09:00:00")
        storage.save_related(mid, [{"id": early, "title": "t", "similarity": 0.9}])
        storage.save_related(late, [{"id": mid, "title": "t", "similarity": 0.9}])
        ev = aggregate_events(storage)[0]
        assert [x["id"] for x in ev["timeline"]] == [early, mid, late]
        assert ev["title"] == "最早报道"

    def test_singleton_excluded_by_default(self, storage):
        _insert(storage, "https://e.com/solo", "孤立报道")
        assert aggregate_events(storage) == []

    def test_singleton_included_with_min_size_1(self, storage):
        aid = _insert(storage, "https://e.com/solo", "孤立报道")
        events = aggregate_events(storage, min_size=1)
        assert len(events) == 1 and events[0]["size"] == 1 and events[0]["id"] == aid

    def test_days_filter_excludes_old_articles(self, storage):
        from datetime import datetime, timedelta

        old = _insert(
            storage, "https://e.com/old", "旧报道",
            fetched_at=(datetime.now() - timedelta(days=10)).isoformat(timespec="seconds"),
        )
        new = _insert(
            storage, "https://e.com/new", "新报道",
            fetched_at=datetime.now().isoformat(timespec="seconds"),
        )
        storage.save_related(new, [{"id": old, "title": "t", "similarity": 0.9}])
        assert aggregate_events(storage, days=3) == []  # 旧文章不在窗口，边被忽略
        assert len(aggregate_events(storage)) == 1  # 全量窗口则成链


class TestConnectionCompat:
    def test_works_with_raw_sqlite_connection(self, tmp_path):
        db = str(tmp_path / "raw.db")
        s = Storage(db)
        a = _insert(s, "https://e.com/a", "A")
        b = _insert(s, "https://e.com/b", "B")
        s.save_related(b, [{"id": a, "title": "A", "similarity": 0.9}])
        s.close()
        conn = sqlite3.connect(db)  # row_factory 默认 None，函数内部应自动补上
        try:
            events = aggregate_events(conn)
            assert len(events) == 1 and events[0]["size"] == 2
        finally:
            conn.close()


class TestMaxSizeConstraint:
    def test_oversized_cluster_capped_at_max_size(self, storage):
        ids = [_insert(storage, f"https://e.com/{i}", f"新闻报道{i}") for i in range(20)]
        for i in range(1, 20):
            storage.save_related(ids[i], [{"id": ids[i - 1], "title": f"新闻报道{i-1}", "similarity": 0.9}])

        events_default = aggregate_events(storage)
        assert len(events_default) == 1
        assert events_default[0]["size"] == 15
        assert len(events_default[0]["articles"]) == 15
        assert len(events_default[0]["timeline"]) == 15

        events_custom = aggregate_events(storage, max_size=5)
        assert len(events_custom) == 1
        assert events_custom[0]["size"] == 5
        assert len(events_custom[0]["articles"]) == 5

    def test_density_prioritization_in_max_size(self, storage):
        hub = _insert(storage, "https://e.com/hub", "核心事件报道", fetched_at="2026-01-01 10:00:00")
        satellites = [_insert(storage, f"https://e.com/sat{i}", f"卫星报道{i}", fetched_at=f"2026-01-01 1{i}:00:00") for i in range(4)]
        for sat in satellites:
            storage.save_related(sat, [{"id": hub, "title": "核心事件报道", "similarity": 0.95}])

        events = aggregate_events(storage, max_size=3)
        assert len(events) == 1
        assert events[0]["size"] == 3
        included_ids = {a["id"] for a in events[0]["articles"]}
        assert hub in included_ids

