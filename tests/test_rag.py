"""rag 模块单元测试：embedding BLOB 存取、余弦相似度、阈值过滤、Top-K、dry-run。"""
import json
import types
from datetime import datetime
from unittest import mock

import numpy as np
import pytest

from src.rag import RAG, EmbeddingConfig, _from_blob, _to_blob, cosine
from src.storage import Storage


def _emb_cfg(**over):
    cfg = {
        "embedding": {
            "similarity_threshold": 0.6,
            "top_k": 3,
            "api_key_env": "TEST_EMB_KEY",
        }
    }
    cfg["embedding"].update(over)
    return EmbeddingConfig(cfg)


def _vec(*vals):
    return np.array(vals, dtype=np.float32)


@pytest.fixture
def storage(tmp_path):
    s = Storage(str(tmp_path / "test.db"))
    yield s
    s.close()


def _insert(storage, url, title="标题", fetched_at="2026-01-01 08:00:00", **kw):
    storage.conn.execute(
        """INSERT INTO articles (url, title, summary, source, published, fetched_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (url, title, kw.get("summary", ""), "测试源", "", fetched_at),
    )
    storage.conn.commit()
    return storage.conn.execute(
        "SELECT id FROM articles WHERE url = ?", (url,)
    ).fetchone()["id"]


class TestBlob:
    def test_roundtrip_preserves_values(self):
        v = _vec(0.1, -0.2, 0.3, 1.0)
        np.testing.assert_allclose(_from_blob(_to_blob(v)), v)

    def test_dtype_is_float32(self):
        assert _from_blob(_to_blob(_vec(1.0, 2.0))).dtype == np.float32

    def test_empty_vector_roundtrip(self):
        out = _from_blob(_to_blob(np.array([], dtype=np.float32)))
        assert out.shape == (0,)
        assert out.dtype == np.float32


class TestCosine:
    def test_identical_vectors_similarity_one(self):
        v = _vec(1, 2, 3)
        np.testing.assert_allclose(cosine(v, np.stack([v])), [1.0], atol=1e-6)

    def test_orthogonal_vectors_zero(self):
        q = _vec(1, 0)
        np.testing.assert_allclose(cosine(q, np.stack([_vec(0, 1)])), [0.0], atol=1e-6)

    def test_zero_vector_does_not_divide_by_zero(self):
        sims = cosine(_vec(0, 0), np.stack([_vec(1, 0)]))
        assert np.isfinite(sims).all()

    def test_similarity_ordering(self):
        q = _vec(1, 0)
        m = np.stack([_vec(0.7, 0.7), _vec(1, 0), _vec(-1, 0)])
        sims = cosine(q, m)
        assert sims[1] > sims[0] > sims[2]


class TestFindRelated:
    def _seed(self, storage):
        vecs = [_vec(1, 0), _vec(0, 1), _vec(0.9, 0.1)]
        ids = []
        for v in vecs:
            aid = _insert(storage, f"https://e.com/h{len(ids)}")
            storage.save_embedding(aid, _to_blob(v))
            ids.append(aid)
        return ids

    def test_filters_below_threshold(self, storage):
        ids = self._seed(storage)
        cur = _insert(storage, "https://e.com/cur")
        rag = RAG(_emb_cfg(similarity_threshold=0.8), "电商", dry_run=True)
        related = rag.find_related(storage, cur, _vec(1, 0))
        assert [r["id"] for r in related] == [ids[0], ids[2]]
        assert all(r["similarity"] >= 0.8 for r in related)

    def test_top_k_limits_results(self, storage):
        ids = self._seed(storage)
        cur = _insert(storage, "https://e.com/cur")
        rag = RAG(_emb_cfg(similarity_threshold=0.0, top_k=2), "电商", dry_run=True)
        related = rag.find_related(storage, cur, _vec(1, 0))
        assert [r["id"] for r in related] == [ids[0], ids[2]]

    def test_empty_history_returns_empty(self, storage):
        rag = RAG(_emb_cfg(), "电商", dry_run=True)
        assert rag.find_related(storage, 1, _vec(1, 0)) == []

    def test_excludes_current_and_later_articles(self, storage):
        ids = self._seed(storage)
        cur = _insert(storage, "https://e.com/cur")
        storage.save_embedding(cur, _to_blob(_vec(1, 0)))
        rag = RAG(_emb_cfg(similarity_threshold=0.0, top_k=10), "电商", dry_run=True)
        related = rag.find_related(storage, cur, _vec(1, 0))
        out_ids = [r["id"] for r in related]
        assert cur not in out_ids
        assert set(out_ids) == set(ids)


class TestVectorize:
    def test_dry_run_returns_count_without_api(self, storage, monkeypatch):
        monkeypatch.delenv("TEST_EMB_KEY", raising=False)
        _insert(storage, "https://e.com/1")
        rag = RAG(_emb_cfg(), "电商", dry_run=True)
        assert rag.vectorize_pending(storage, dry_run=True) == 1
        assert storage.articles_without_embedding()  # dry-run 不落库

    def test_saves_blobs(self, storage, monkeypatch):
        monkeypatch.setenv("TEST_EMB_KEY", "sk-test")
        _insert(storage, "https://e.com/1", title="A")
        _insert(storage, "https://e.com/2", title="B")
        rag = RAG(_emb_cfg(), "电商")
        resp = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(embedding=[0.1, 0.2]),
                types.SimpleNamespace(embedding=[0.3, 0.4]),
            ]
        )
        rag.emb_client.embeddings.create = mock.Mock(return_value=resp)
        assert rag.vectorize_pending(storage) == 2
        rows = storage.embedded_articles()
        assert len(rows) == 2
        for r in rows:
            assert _from_blob(r["embedding"]).dtype == np.float32
        rag.emb_client.embeddings.create.assert_called_once()

    def test_batch_chunking_and_retry(self, storage, monkeypatch):
        monkeypatch.setenv("TEST_EMB_KEY", "sk-test")
        for i in range(5):
            _insert(storage, f"https://e.com/{i}", title=f"文章{i}")
        rag = RAG(_emb_cfg(), "电商", batch_size=2)

        def make_resp(texts):
            return types.SimpleNamespace(
                data=[types.SimpleNamespace(embedding=[0.1 * idx, 0.2 * idx]) for idx, _ in enumerate(texts)]
            )

        call_count = 0

        def mocked_create(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            texts = kwargs.get("input", [])
            # 模拟第 2 批次第 1 次调用时网络抖动超时，随后重试成功
            if call_count == 2:
                raise RuntimeError("临时网络抖动")
            return make_resp(texts)

        rag.emb_client.embeddings.create = mock.Mock(side_effect=mocked_create)
        with mock.patch("src.rag.time.sleep"):
            count = rag.vectorize_pending(storage)

        assert count == 5
        # 5 篇文章按 batch_size=2，分为 3 批：第 1 批成功(1次)，第 2 批失败后重试成功(2次)，第 3 批成功(1次)，共 4 次调用
        assert call_count == 4
        assert len(storage.embedded_articles()) == 5


class TestRelate:
    def test_dry_run_counts_pending_without_saving(self, storage):
        aid = _insert(storage, "https://e.com/1")
        storage.save_embedding(aid, _to_blob(_vec(1, 0)))
        rag = RAG(_emb_cfg(), "电商", dry_run=True)
        assert rag.relate_pending(storage, dry_run=True) == 1
        raw = storage.conn.execute(
            "SELECT related_event FROM articles WHERE id = ?", (aid,)
        ).fetchone()["related_event"]
        assert raw is None

    def test_saves_related_for_pending(self, storage):
        aid = _insert(storage, "https://e.com/1")
        storage.save_embedding(aid, _to_blob(_vec(1, 0)))
        rag = RAG(_emb_cfg(similarity_threshold=0.0), "电商", dry_run=True)
        assert rag.relate_pending(storage) == 1
        raw = storage.conn.execute(
            "SELECT related_event FROM articles WHERE id = ?", (aid,)
        ).fetchone()["related_event"]
        assert json.loads(raw) == []  # 无历史文章，空关联

    def test_skips_already_related(self, storage):
        aid = _insert(storage, "https://e.com/1")
        storage.save_embedding(aid, _to_blob(_vec(1, 0)))
        storage.save_related(aid, [])
        rag = RAG(_emb_cfg(), "电商", dry_run=True)
        assert rag.relate_pending(storage) == 0  # 幂等：已关联跳过


class TestWeeklyTrends:
    def _seed_cluster(self, storage):
        now = datetime.now().isoformat(timespec="seconds")
        a = _insert(storage, "https://e.com/1", title="事件一", fetched_at=now)
        b = _insert(storage, "https://e.com/2", title="事件二", fetched_at=now)
        storage.save_related(a, [{"id": b, "title": "事件二", "similarity": 0.9}])
        storage.save_related(b, [{"id": a, "title": "事件一", "similarity": 0.9}])
        return a, b

    def test_dry_run_returns_empty_and_no_save(self, storage):
        self._seed_cluster(storage)
        rag = RAG(_emb_cfg(), "电商", dry_run=True)
        assert rag.weekly_trends(storage, dry_run=True) == []
        n = storage.conn.execute("SELECT COUNT(*) AS n FROM trends").fetchone()["n"]
        assert n == 0

    def test_fallback_joins_titles_without_chat_client(self, storage, monkeypatch):
        monkeypatch.setenv("TEST_EMB_KEY", "sk-test")
        self._seed_cluster(storage)
        rag = RAG(_emb_cfg(), "电商")  # llm_cfg=None → chat_client 为空，走降级拼接
        results = rag.weekly_trends(storage)
        assert len(results) == 1
        assert set(results[0]["content"].split("；")) == {"事件一", "事件二"}
        rows = storage.latest_trends()
        assert len(rows) == 1
        assert json.loads(rows[0]["cluster"])

    def test_singleton_article_skipped_in_weekly_trends(self, storage, monkeypatch):
        monkeypatch.setenv("TEST_EMB_KEY", "sk-test")
        now = datetime.now().isoformat(timespec="seconds")
        a = _insert(storage, "https://e.com/single", title="孤立事件", fetched_at=now)
        storage.save_related(a, [])  # 无历史相似文章，孤立单篇
        rag = RAG(_emb_cfg(), "电商")
        results = rag.weekly_trends(storage)
        assert results == []
        assert storage.conn.execute("SELECT COUNT(*) AS n FROM trends").fetchone()["n"] == 0

    def test_chat_completion_exception_falls_back_to_title_join(self, storage, monkeypatch):
        monkeypatch.setenv("TEST_EMB_KEY", "sk-test")
        self._seed_cluster(storage)
        mock_llm_cfg = mock.MagicMock()
        mock_llm_cfg.available = True
        mock_llm_cfg.model = "test-model"
        rag = RAG(_emb_cfg(), "电商", llm_cfg=mock_llm_cfg, dry_run=True)
        rag.chat_client = mock.MagicMock()
        rag.chat_client.chat.completions.create.side_effect = Exception("API 500 Error")
        results = rag.weekly_trends(storage)
        assert len(results) == 1
        assert set(results[0]["content"].split("；")) == {"事件一", "事件二"}
        rows = storage.latest_trends()
        assert len(rows) == 1
        assert json.loads(rows[0]["cluster"])

    def test_oversized_trend_cluster_capped_at_limit(self, storage, monkeypatch):
        monkeypatch.setenv("TEST_EMB_KEY", "sk-test")
        now = datetime.now().isoformat(timespec="seconds")
        ids = [_insert(storage, f"https://e.com/trend/{i}", title=f"趋势文章{i}", fetched_at=now) for i in range(15)]
        for i in range(1, 15):
            storage.save_related(ids[i], [{"id": ids[i - 1], "title": f"趋势文章{i-1}", "similarity": 0.9}])
        rag = RAG(_emb_cfg(), "电商")
        results = rag.weekly_trends(storage, max_cluster_size=5)
        assert len(results) == 1
        assert len(results[0]["cluster"]) == 5
        assert len(results[0]["content"].split("；")) == 5


class TestRun:
    def test_run_dry_run_stats(self, storage, monkeypatch):
        monkeypatch.delenv("TEST_EMB_KEY", raising=False)
        _insert(storage, "https://e.com/1")
        _insert(storage, "https://e.com/2")
        rag = RAG(_emb_cfg(), "电商", dry_run=True)
        assert rag.run(storage, dry_run=True) == {"vectorized": 2, "related": 0, "trends": 0}
