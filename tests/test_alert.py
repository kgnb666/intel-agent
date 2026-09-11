"""alert 模块单元测试：候选筛选、payload 构造、webhook 推送与失败降级。"""
from unittest import mock

import requests

from src.alert import (
    AlertConfig,
    alert_candidates,
    build_payload,
    filter_unsent_alerts,
    push_alert,
    push_alerts,
)


def _event(size=2, sentiment="负面", conf=0.9, title="某公司股价暴跌"):
    return {
        "id": 1,
        "title": title,
        "size": size,
        "articles": [
            {
                "id": i,
                "title": f"报道{i}",
                "url": f"https://e.com/{i}",
                "source": "源A",
                "fetched_at": f"2026-01-0{i} 08:00:00",
                "sentiment": sentiment,
                "sentiment_conf": conf,
            }
            for i in range(1, size + 1)
        ],
        "timeline": [
            {
                "id": i,
                "date": f"2026-01-0{i}",
                "title": f"报道{i}",
                "source": "源A",
                "url": f"https://e.com/{i}",
                "sentiment": sentiment,
            }
            for i in range(1, size + 1)
        ],
    }


class TestAlertConfig:
    def test_unavailable_without_webhook(self):
        assert AlertConfig({}).available is False
        assert AlertConfig({"alert": {"webhook": ""}}).available is False

    def test_webhook_from_config(self):
        cfg = AlertConfig({"alert": {"webhook": "https://cfg.example.com/hook"}})
        assert cfg.available is True
        assert cfg.webhook == "https://cfg.example.com/hook"

    def test_cli_overrides_config(self):
        cfg = AlertConfig(
            {"alert": {"webhook": "https://cfg.example.com/hook"}},
            webhook="https://cli.example.com/hook",
        )
        assert cfg.webhook == "https://cli.example.com/hook"

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("ALERT_WEBHOOK", "https://env.example.com/hook")
        assert AlertConfig({}).webhook == "https://env.example.com/hook"

    def test_default_min_confidence(self):
        assert AlertConfig({}).min_confidence == 0.8


class TestAlertCandidates:
    def test_negative_high_conf_selected(self):
        assert alert_candidates([_event(conf=0.9)], 0.8) == [_event(conf=0.9)]

    def test_negative_low_conf_excluded(self):
        assert alert_candidates([_event(conf=0.5)], 0.8) == []

    def test_non_negative_excluded(self):
        assert alert_candidates([_event(sentiment="正面", conf=0.95)], 0.8) == []
        assert alert_candidates([_event(sentiment="中性", conf=0.95)], 0.8) == []

    def test_max_conf_across_articles(self):
        ev = _event(size=2, conf=0.7)
        ev["articles"][1]["sentiment_conf"] = 0.9
        assert alert_candidates([ev], 0.8) == [ev]


class TestBuildPayload:
    def test_dingtalk_format_uses_text(self):
        payload = build_payload(
            "https://oapi.dingtalk.com/robot/send?access_token=x", _event()
        )
        assert payload["msgtype"] == "markdown"
        assert "text" in payload["markdown"]
        assert "content" not in payload["markdown"]

    def test_wecom_format_uses_content(self):
        payload = build_payload(
            "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=x", _event()
        )
        assert "content" in payload["markdown"]
        assert "text" not in payload["markdown"]

    def test_payload_contains_title_confidence_timeline(self):
        payload = build_payload(
            "https://oapi.dingtalk.com/robot/send?access_token=x",
            _event(size=2, conf=0.93),
            industry="电商",
        )
        text = payload["markdown"]["text"]
        assert "负面事件告警：某公司股价暴跌" in text
        assert "0.93" in text
        assert "涉及行业：电商" in text
        assert "时间线" in text
        assert "2026-01-01" in text


class TestPushAlert:
    def test_success_returns_true(self):
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"errcode": 0}
        with mock.patch("src.alert.requests.post", return_value=resp) as post:
            assert push_alert("https://hook", {"msgtype": "markdown"}) is True
        post.assert_called_once()
        assert post.call_args.kwargs["json"]["msgtype"] == "markdown"

    def test_network_error_returns_false(self):
        with mock.patch(
            "src.alert.requests.post", side_effect=requests.ConnectionError("boom")
        ):
            assert push_alert("https://hook", {}) is False

    def test_http_error_returns_false(self):
        resp = mock.Mock(status_code=500, text="Internal Server Error")
        with mock.patch("src.alert.requests.post", return_value=resp):
            assert push_alert("https://hook", {}) is False

    def test_nonzero_errcode_returns_false(self):
        resp = mock.Mock(status_code=200, text='{"errcode": 93000}')
        resp.json.return_value = {"errcode": 93000}
        with mock.patch("src.alert.requests.post", return_value=resp):
            assert push_alert("https://hook", {}) is False

    def test_non_json_2xx_returns_true(self):
        resp = mock.Mock(status_code=200)
        resp.json.side_effect = ValueError("no json")
        with mock.patch("src.alert.requests.post", return_value=resp):
            assert push_alert("https://hook", {}) is True

    def test_nonzero_errcode_logs_diagnostic_warning(self, caplog):
        resp = mock.Mock(
            status_code=200,
            text='{"errcode": 130101, "errmsg": "IP不在白名单"}',
        )
        resp.json.return_value = {"errcode": 130101, "errmsg": "IP不在白名单"}
        with mock.patch("src.alert.requests.post", return_value=resp):
            with caplog.at_level("WARNING", logger="alert"):
                ok = push_alert("https://hook", {"msgtype": "markdown"})
                assert ok is False
                assert "130101" in caplog.text
                assert "IP不在白名单" in caplog.text

    def test_http_error_logs_diagnostic_warning(self, caplog):
        resp = mock.Mock(status_code=502, text="Bad Gateway Error from proxy")
        with mock.patch("src.alert.requests.post", return_value=resp):
            with caplog.at_level("WARNING", logger="alert"):
                ok = push_alert("https://hook", {"msgtype": "markdown"})
                assert ok is False
                assert "HTTP 502" in caplog.text
                assert "Bad Gateway" in caplog.text


class TestPushAlerts:
    def test_partial_failure_counts_success(self):
        evs = [_event(size=2, title=f"事件{i}") for i in range(2)]
        with mock.patch("src.alert.push_alert", side_effect=[True, False]) as push:
            assert push_alerts("https://hook", evs) == 1
        assert push.call_count == 2

    def test_all_fail_returns_zero_without_raising(self):
        with mock.patch("src.alert.push_alert", return_value=False):
            assert push_alerts("https://hook", [_event()]) == 0

    def test_push_alerts_records_sent_status_with_storage(self):
        storage = mock.Mock()
        ev = _event(size=2, conf=0.95)
        ev["id"] = 10
        ev["articles"][0]["id"] = 101
        with mock.patch("src.alert.push_alert", return_value=True):
            sent = push_alerts("https://hook", [ev], storage=storage)
            assert sent == 1
        storage.record_alert_sent.assert_called_once_with(10, 101)


class TestFilterUnsentAlerts:
    def test_none_storage_returns_all(self):
        ev = _event()
        assert filter_unsent_alerts([ev], None) == [ev]

    def test_filters_already_sent_alerts(self):
        ev1 = _event(title="事件1")
        ev1["id"] = 1
        ev1["articles"][0]["id"] = 11
        ev2 = _event(title="事件2")
        ev2["id"] = 2
        ev2["articles"][0]["id"] = 22

        storage = mock.Mock()
        storage.is_alert_sent.side_effect = lambda eid, aid: eid == 1
        res = filter_unsent_alerts([ev1, ev2], storage)
        assert len(res) == 1
        assert res[0]["id"] == 2

    def test_extracts_highest_negative_article_id(self):
        ev = {
            "id": 5,
            "title": "事件5",
            "articles": [
                {"id": 501, "sentiment": "正面", "sentiment_conf": 0.9},
                {"id": 502, "sentiment": "负面", "sentiment_conf": 0.85},
                {"id": 503, "sentiment": "负面", "sentiment_conf": 0.98},
            ],
        }
        storage = mock.Mock()
        storage.is_alert_sent.return_value = False
        res = filter_unsent_alerts([ev], storage)
        assert len(res) == 1
        storage.is_alert_sent.assert_called_once_with(5, 503)

