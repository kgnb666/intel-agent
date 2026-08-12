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


# ---------- HTML 清洗 ----------

class TestCleanText:
    def test_strips_tags_and_collapses_space(self):
        html = "<p>拼多多  <b>财报</b></p>\n\n<span>超预期</span>"
        assert crawler._clean_text(html) == "拼多多 财报 超预期"


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
