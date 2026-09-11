"""crawler 模块单元测试：关键词过滤、相似度去重、HTML 清洗、重试机制。"""
import os
import sys
import types
from unittest import mock

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import crawler


# ---------- 关键词过滤 ----------

class TestKeywordMatch:
    KEYWORDS = ["电商", "拼多多", "消费"]

    def test_hit_single_keyword(self):
        assert crawler._match_keywords("拼多多发布 Q2 财报", self.KEYWORDS) is True

    def test_hit_in_summary_not_title(self):
        text = "某科技公司裁员 " + "本地生活与电商业务收缩"
        assert crawler._match_keywords(text, self.KEYWORDS) is True

    def test_no_keyword_returns_false(self):
        assert crawler._match_keywords("纯硬件评测：新款显卡跑分", self.KEYWORDS) is False

    def test_empty_keywords_matches_nothing(self):
        # 防御式校验：空关键词列表应全过滤，避免误采集无关内容
        assert crawler._match_keywords("任何文本", []) is False

    def test_case_insensitive_match_iphone(self):
        # 验证 keyword 为 "iPhone" 时，能正确命中包含 "iphone 发布会" 的文本
        assert crawler._match_keywords("全新 iphone 发布会定档下周", ["iPhone"]) is True

    def test_case_insensitive_match_ai(self):
        # 验证 keyword 为 "AI" 时，能正确命中包含 "ai 应用落地" 的文本
        assert crawler._match_keywords("探讨电商行业 ai 应用落地新范式", ["AI"]) is True


# ---------- 标题相似度去重 ----------

class TestTitleSimilarity:
    THRESHOLD = 0.85

    def test_same_event_different_source_is_duplicate(self):
        existing = ["拼多多发布 2026 年第二季度财报，营收超预期"]
        title = "拼多多发布2026年第二季度财报，营收超预期！"
        assert crawler._is_similar(title, existing, self.THRESHOLD) is True

    def test_different_event_not_duplicate(self):
        existing = ["拼多多发布 2026 年第二季度财报"]
        title = "京东宣布百亿补贴全面升级"
        assert crawler._is_similar(title, existing, self.THRESHOLD) is False

    def test_empty_existing_never_duplicate(self):
        assert crawler._is_similar("任意标题", [], self.THRESHOLD) is False

    def test_empty_or_blank_strings(self):
        assert crawler._is_similar("", ["已有标题"], self.THRESHOLD) is False
        assert crawler._is_similar("新标题", [""], self.THRESHOLD) is False

    def test_length_pruning_fast_reject(self):
        # 长度差异极大：2 * 5 / (5 + 30) = 0.285 < 0.85
        short_title = "快讯通知"
        long_existing = ["快讯通知：" + "这是一条非常长非常长非常长的行业快讯详细新闻" * 3]
        with mock.patch("src.crawler.SequenceMatcher") as sm_mock:
            assert crawler._is_similar(short_title, long_existing, self.THRESHOLD) is False
            sm_mock.assert_not_called()

    def test_char_overlap_below_threshold_fast_reject(self):
        # 字符集重叠率低于 0.6：直接 fast reject，不触发 SequenceMatcher
        title = "abcdefghij"
        existing = ["abklmnopqr"]  # 仅 'a', 'b' 重合，2 / 10 = 0.2 < 0.6
        with mock.patch("src.crawler.SequenceMatcher") as sm_mock:
            assert crawler._is_similar(title, existing, self.THRESHOLD) is False
            sm_mock.assert_not_called()

    def test_char_overlap_above_threshold_triggers_sequence_matcher(self):
        # 字符集重叠率 >= 0.6：通过初筛，调用 SequenceMatcher
        # 10 个字符中重合 7 个：7 / 10 = 0.7 >= 0.6
        title = "abcdefg123"
        existing = ["abcdefg890"]
        with mock.patch("src.crawler.SequenceMatcher") as sm_mock:
            sm_mock.return_value.ratio.return_value = 0.7
            res = crawler._is_similar(title, existing, self.THRESHOLD)
            sm_mock.assert_called_once()
            assert res is False

    def test_high_char_overlap_different_sequence_is_not_duplicate(self):
        # 字符集合高度重叠，但顺序不同或语义不匹配（SequenceMatcher ratio < 0.85）
        title = "张三收购了李四的企业"
        existing = ["李四的企业收购了张三"]
        # 字符集完全一样，但 SequenceMatcher.ratio() 不会达到 0.85
        assert crawler._is_similar(title, existing, self.THRESHOLD) is False


# ---------- HTML 清洗 ----------

class TestCleanText:
    def test_strips_tags_and_collapses_space(self):
        html = "<p>拼多多  <b>财报</b></p>\n\n<span>超预期</span>"
        assert crawler._clean_text(html) == "拼多多 财报 超预期"

    def test_unescapes_html_entities(self):
        raw = '<p>拼多多&nbsp;发布&quot;财报&quot;&amp;升级</p>'
        assert crawler._clean_text(raw) == '拼多多 发布"财报"&升级'


# ---------- 请求重试机制 ----------

class TestGetWithRetry:
    def test_retries_then_succeeds(self):
        ok = types.SimpleNamespace(raise_for_status=lambda: None, content=b"x")
        with mock.patch.object(
            crawler.requests, "get",
            side_effect=[requests.ConnectionError("boom"), ok],
        ) as m, mock.patch.object(crawler.time, "sleep") as s:
            resp = crawler._get_with_retry("http://x", retries=2, interval=5)
        assert resp is ok and m.call_count == 2
        s.assert_called_once_with(5)

    def test_exhausts_retries_and_raises(self):
        with mock.patch.object(
            crawler.requests, "get",
            side_effect=requests.Timeout("slow"),
        ) as m, mock.patch.object(crawler.time, "sleep"):
            with pytest.raises(requests.Timeout):
                crawler._get_with_retry("http://x", retries=2, interval=5)
        assert m.call_count == 3  # 首次 + 2 次重试


# ---------- 采集流水线：熔断与批量入库 ----------

class TestRunCrawlCircuitBreakerAndBatch:
    def test_broken_feed_is_skipped(self):
        cfg = {
            "industry": {"keywords": ["电商"]},
            "crawl": {"max_items_per_source": 10, "title_similarity_threshold": 0.85},
            "sources": [{"name": "故障源", "url": "https://broken.com/rss"}],
        }
        storage = mock.Mock()
        storage.recent_titles.return_value = []
        storage.is_feed_broken.return_value = True

        with mock.patch("src.crawler.fetch_rss") as fetch_mock:
            stats = crawler.run_crawl(cfg, storage)
            fetch_mock.assert_not_called()

        assert any("[熔断跳过]" in err for err in stats["errors"])
        assert stats["fetched"] == 0

    def test_batch_insertion_called_with_matched_items(self):
        cfg = {
            "industry": {"keywords": ["电商"]},
            "crawl": {"max_items_per_source": 10, "title_similarity_threshold": 0.85},
            "sources": [{"name": "正常源", "url": "https://ok.com/rss"}],
        }
        storage = mock.Mock()
        storage.recent_titles.return_value = []
        storage.is_feed_broken.return_value = False
        storage.insert_articles.return_value = 2

        items = [
            {"title": "电商行业季度销售数据出炉", "summary": "摘要1", "url": "https://ok.com/1"},
            {"title": "电商平台宣布降低中小商家技术服务费率", "summary": "摘要2", "url": "https://ok.com/2"},
            {"title": "无关硬件显卡芯片评测资讯", "summary": "摘要3", "url": "https://ok.com/3"},
        ]
        with mock.patch("src.crawler.fetch_rss", return_value=items):
            stats = crawler.run_crawl(cfg, storage)

        assert stats["fetched"] == 3
        assert stats["matched"] == 2
        assert stats["inserted"] == 2
        storage.insert_articles.assert_called_once()
        inserted_items = storage.insert_articles.call_args[0][0]
        assert len(inserted_items) == 2
        storage.record_feed_result.assert_called_once_with("https://ok.com/rss", "正常源", success=True)

