"""analyzer 模块单元测试：prompt 构造、JSON 解析容错、重试次数、dry-run。"""
import json
import types
from unittest import mock

import pytest

from src.analyzer import Analyzer, LLMConfig, _extract_json, build_prompt

VALID = {
    "summary": "一句话摘要",
    "sentiment": "正面",
    "confidence": 0.8,
    "tags": ["电商"],
    "entities": ["拼多多"],
}


def _response(content, tokens=10):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))],
        usage=types.SimpleNamespace(total_tokens=tokens),
    )


def _llm_cfg(**over):
    cfg = {
        "llm": {
            "base_url": "https://x",
            "model": "test-model",
            "api_key_env": "TEST_LLM_KEY",
            "max_per_run": 3,
            "temperature": 0.1,
        }
    }
    cfg["llm"].update(over)
    return LLMConfig(cfg)


class TestBuildPrompt:
    def test_contains_industry_and_article_fields(self):
        p = build_prompt("电商", {"title": "拼多多财报", "summary": "营收超预期"})
        assert "电商" in p
        assert "拼多多财报" in p
        assert "营收超预期" in p
        for field in ("summary", "sentiment", "confidence", "tags", "entities"):
            assert field in p

    def test_prefers_summary_as_content(self):
        p = build_prompt("电商", {"title": "标题", "summary": "摘要内容"})
        assert "摘要内容" in p

    def test_falls_back_to_title_without_summary(self):
        p = build_prompt("电商", {"title": "只有标题"})
        assert "只有标题" in p


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json(json.dumps(VALID, ensure_ascii=False)) == VALID

    def test_code_fence_and_noise(self):
        text = "好的，结果如下：\n```json\n" + json.dumps(VALID, ensure_ascii=False) + "\n```\n希望对你有帮助"
        assert _extract_json(text)["summary"] == VALID["summary"]

    def test_bad_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json("这不是 JSON {坏了")

    def test_missing_field_raises(self):
        bad = dict(VALID)
        bad.pop("tags")
        with pytest.raises(ValueError, match="缺少字段"):
            _extract_json(json.dumps(bad, ensure_ascii=False))

    def test_invalid_sentiment_raises(self):
        bad = dict(VALID, sentiment="开心")
        with pytest.raises(ValueError, match="非法情感值"):
            _extract_json(json.dumps(bad, ensure_ascii=False))

    def test_no_json_object_raises(self):
        with pytest.raises(ValueError, match="未找到"):
            _extract_json("完全没有大括号")

    def test_confidence_clamped_to_unit_range(self):
        high = _extract_json(json.dumps(dict(VALID, confidence=3.5)))
        low = _extract_json(json.dumps(dict(VALID, confidence=-1)))
        assert high["confidence"] == 1.0
        assert low["confidence"] == 0.0


class TestAnalyzerDryRun:
    def test_dry_run_does_not_require_client_or_api(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        a = Analyzer(_llm_cfg(), "电商", dry_run=True)
        assert a.client is None
        out = a.analyze_one({"id": 1, "title": "T", "summary": "S"}, dry_run=True)
        assert out["ok"] is True
        assert out["dry_run"] is True
        assert "T" in out["prompt"]
        assert out["tokens"] == 0

    def test_run_dry_run_counts_and_never_saves(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        storage = mock.Mock()
        storage.unanalyzed_articles.return_value = [
            {"id": 1, "title": "A", "summary": "a"},
            {"id": 2, "title": "B", "summary": "b"},
        ]
        a = Analyzer(_llm_cfg(), "电商", dry_run=True)
        stats = a.run(storage, dry_run=True)
        assert stats == {"total": 2, "success": 2, "failed": 0, "tokens": 0, "errors": []}
        storage.save_analysis.assert_not_called()


class TestAnalyzerApi:
    def test_success_parses_result_and_tokens(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = _response(
            json.dumps(VALID, ensure_ascii=False), tokens=17
        )
        a = Analyzer(_llm_cfg(), "电商", client=client)
        out = a.analyze_one({"id": 1, "title": "T", "summary": "S"})
        assert out["ok"] is True
        assert out["tokens"] == 17
        assert out["result"]["sentiment"] == "正面"
        client.chat.completions.create.assert_called_once()

    def test_parse_failure_retries_twice_then_fails(self):
        client = mock.Mock()
        client.chat.completions.create.return_value = _response("不是 JSON")
        a = Analyzer(_llm_cfg(), "电商", client=client)
        out = a.analyze_one({"id": 1, "title": "T", "summary": "S"})
        assert out["ok"] is False
        assert "JSON" in out["error"]
        assert client.chat.completions.create.call_count == 2  # 解析失败最多重试 2 次

    def test_run_accumulates_stats_and_saves(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = [
            _response(json.dumps(VALID, ensure_ascii=False), tokens=5),
            _response("坏输出"),
            _response("坏输出"),
            _response(json.dumps(VALID, ensure_ascii=False), tokens=7),
        ]
        storage = mock.Mock()
        storage.unanalyzed_articles.return_value = [
            {"id": 1, "title": "A", "summary": "a"},
            {"id": 2, "title": "B", "summary": "b"},
            {"id": 3, "title": "C", "summary": "c"},
        ]
        a = Analyzer(_llm_cfg(max_per_run=3), "电商", client=client)
        stats = a.run(storage)
        assert stats["total"] == 3
        assert stats["success"] == 2
        assert stats["failed"] == 1
        assert stats["tokens"] == 12
        assert len(stats["errors"]) == 1
        assert storage.save_analysis.call_count == 2


class TestBackoff:
    def test_retries_transient_error_then_succeeds(self):
        ok = _response(json.dumps(VALID, ensure_ascii=False))
        client = mock.Mock()
        client.chat.completions.create.side_effect = [RuntimeError("boom"), ok]
        a = Analyzer(_llm_cfg(), "电商", client=client)
        with mock.patch("src.analyzer.time.sleep") as sleep:
            resp = a._call_with_backoff("p")
        assert resp is ok
        assert client.chat.completions.create.call_count == 2
        sleep.assert_called_once_with(2)

    def test_exhausts_retries_and_raises(self):
        client = mock.Mock()
        client.chat.completions.create.side_effect = RuntimeError("boom")
        a = Analyzer(_llm_cfg(), "电商", client=client)
        with mock.patch("src.analyzer.time.sleep"):
            with pytest.raises(RuntimeError):
                a._call_with_backoff("p")
        assert client.chat.completions.create.call_count == 3  # 首次 + 2 次退避重试


class TestLLMConfig:
    def test_env_overrides_config(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://env.example.com")
        monkeypatch.setenv("LLM_MODEL", "env-model")
        monkeypatch.setenv("TEST_LLM_KEY", "sk-env")
        c = _llm_cfg()
        assert c.base_url == "https://env.example.com"
        assert c.model == "env-model"
        assert c.api_key == "sk-env"
        assert c.available is True

    def test_unavailable_without_key(self, monkeypatch):
        monkeypatch.delenv("TEST_LLM_KEY", raising=False)
        c = _llm_cfg()
        assert c.available is False

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        c = LLMConfig({"llm": {}})
        assert c.model == "deepseek-chat"
        assert c.max_per_run == 20
        assert c.temperature == 0.2
