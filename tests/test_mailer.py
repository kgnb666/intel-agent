import email
import sys
from unittest import mock

import pytest

import run_daily
from src.config import load_config
from src.mailer import MailConfig, send_html

SMTP_ENV = ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_TO"]


@pytest.fixture
def no_smtp_env(monkeypatch):
    for k in SMTP_ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


@pytest.fixture
def full_smtp_env(no_smtp_env):
    no_smtp_env.setenv("SMTP_HOST", "smtp.example.com")
    no_smtp_env.setenv("SMTP_PORT", "465")
    no_smtp_env.setenv("SMTP_USER", "a@example.com")
    no_smtp_env.setenv("SMTP_PASSWORD", "secret")
    no_smtp_env.setenv("SMTP_TO", "x@example.com, y@example.com")
    return no_smtp_env


class TestMailConfig:
    def test_unavailable_without_smtp_env(self, no_smtp_env):
        assert MailConfig().available is False

    def test_available_with_full_env(self, full_smtp_env):
        cfg = MailConfig()
        assert cfg.available is True
        assert cfg.to == ["x@example.com", "y@example.com"]

    def test_partial_env_unavailable(self, no_smtp_env):
        no_smtp_env.setenv("SMTP_HOST", "smtp.example.com")
        no_smtp_env.setenv("SMTP_USER", "a@example.com")
        no_smtp_env.setenv("SMTP_TO", "x@example.com")
        assert MailConfig().available is False  # 缺 password 视为不可用

    def test_available_with_mail_to_fallback(self, no_smtp_env):
        no_smtp_env.setenv("SMTP_HOST", "smtp.example.com")
        no_smtp_env.setenv("SMTP_PORT", "465")
        no_smtp_env.setenv("SMTP_USER", "a@example.com")
        no_smtp_env.setenv("SMTP_PASSWORD", "secret")
        no_smtp_env.delenv("SMTP_TO", raising=False)
        no_smtp_env.setenv("MAIL_TO", "backup@example.com, other@example.com")
        cfg = MailConfig()
        assert cfg.available is True
        assert cfg.to == ["backup@example.com", "other@example.com"]


class TestSendHtml:
    def test_sends_via_ssl_465(self, full_smtp_env):
        cfg = MailConfig()
        with mock.patch("src.mailer.smtplib.SMTP_SSL") as ssl_cls:
            server = ssl_cls.return_value
            send_html(cfg, "标题", "<html><body>hi</body></html>")
        ssl_cls.assert_called_once_with("smtp.example.com", 465, timeout=30)
        server.login.assert_called_once_with("a@example.com", "secret")
        server.sendmail.assert_called_once()
        assert server.sendmail.call_args[0][1] == ["x@example.com", "y@example.com"]
        server.quit.assert_called_once()

    def test_sends_via_starttls_587(self, no_smtp_env):
        env = {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "a@example.com",
            "SMTP_PASSWORD": "p",
            "SMTP_TO": "x@example.com",
        }
        for k, v in env.items():
            no_smtp_env.setenv(k, v)
        cfg = MailConfig()
        with mock.patch("src.mailer.smtplib.SMTP") as smtp_cls:
            server = smtp_cls.return_value
            send_html(cfg, "标题", "<html/>")
        smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=30)
        server.starttls.assert_called_once()
        server.login.assert_called_once()
        server.sendmail.assert_called_once()
        server.quit.assert_called_once()

    def test_sends_multipart_alternative_with_plain_and_html(self, full_smtp_env):
        cfg = MailConfig()
        with mock.patch("src.mailer.smtplib.SMTP_SSL") as ssl_cls:
            server = ssl_cls.return_value
            send_html(cfg, "今日情报", "<h1>HTML日报</h1>", plain_body="纯文本日报")

        raw_msg = server.sendmail.call_args[0][2]
        msg = email.message_from_string(raw_msg)
        assert msg.is_multipart() is True
        assert msg.get_content_type() == "multipart/alternative"

        parts = msg.get_payload()
        assert len(parts) == 2
        # 第一部分：纯文本
        assert parts[0].get_content_type() == "text/plain"
        assert "纯文本日报" in parts[0].get_payload(decode=True).decode("utf-8")
        # 第二部分：HTML
        assert parts[1].get_content_type() == "text/html"
        assert "<h1>HTML日报</h1>" in parts[1].get_payload(decode=True).decode("utf-8")

    def test_sends_default_plain_fallback_when_omitted(self, full_smtp_env):
        cfg = MailConfig()
        with mock.patch("src.mailer.smtplib.SMTP_SSL") as ssl_cls:
            server = ssl_cls.return_value
            send_html(cfg, "今日情报", "<p>仅提供HTML</p>")

        raw_msg = server.sendmail.call_args[0][2]
        msg = email.message_from_string(raw_msg)
        parts = msg.get_payload()
        assert len(parts) == 2
        assert parts[0].get_content_type() == "text/plain"
        assert "请在支持 HTML 的客户端中查看" in parts[0].get_payload(decode=True).decode("utf-8")


class TestRunDailyNoSend:
    @pytest.fixture
    def isolated(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        for k in SMTP_ENV:
            monkeypatch.delenv(k, raising=False)
        cfg = load_config()
        cfg["storage"]["db_path"] = str(tmp_path / "intel_test.db")
        monkeypatch.setattr(run_daily, "load_config_or_exit", lambda: cfg)
        out_dir = tmp_path / "out"
        monkeypatch.setattr(run_daily, "OUTPUT_DIR", str(out_dir))
        monkeypatch.setattr(sys, "argv", ["run_daily.py", "--no-send", "--day", "2099-12-31"])
        return tmp_path, out_dir

    def test_no_send_writes_html_without_smtp(self, isolated, monkeypatch):
        _, out_dir = isolated
        monkeypatch.setattr(
            run_daily, "run_crawl",
            lambda cfg, storage: {"fetched": 1, "matched": 1, "inserted": 1, "errors": []},
        )
        run_daily.main()
        files = list(out_dir.glob("daily_*.html"))
        assert len(files) == 1
        assert "情报日报" in files[0].read_text(encoding="utf-8")

    def test_degraded_note_written_when_crawl_fails(self, isolated, monkeypatch):
        _, out_dir = isolated

        def boom(cfg, storage):
            raise RuntimeError("网络故障")

        monkeypatch.setattr(run_daily, "run_crawl", boom)
        run_daily.main()
        files = list(out_dir.glob("daily_*.html"))
        assert len(files) == 1
        html = files[0].read_text(encoding="utf-8")
        assert "本次流水线部分环节降级" in html
        assert "采集环节失败" in html
        assert "网络故障" in html


class TestRunDailyWebhook:
    @pytest.fixture
    def seeded(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("ALERT_WEBHOOK", raising=False)
        for k in SMTP_ENV:
            monkeypatch.delenv(k, raising=False)
        cfg = load_config()
        cfg["storage"]["db_path"] = str(tmp_path / "intel_test.db")
        monkeypatch.setattr(run_daily, "load_config_or_exit", lambda: cfg)
        out_dir = tmp_path / "out"
        monkeypatch.setattr(run_daily, "OUTPUT_DIR", str(out_dir))
        monkeypatch.setattr(
            run_daily, "run_crawl",
            lambda cfg, storage: {"fetched": 0, "matched": 0, "inserted": 0, "errors": []},
        )

        from src.storage import Storage

        s = Storage(cfg["storage"]["db_path"])
        s.conn.execute(
            """INSERT INTO articles (url, title, summary, source, published, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("https://e.com/1", "某公司股价暴跌", "", "源A", "", "2099-01-01 08:00:00"),
        )
        s.conn.execute(
            """INSERT INTO articles (url, title, summary, source, published, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("https://e.com/2", "某公司股价暴跌后续", "", "源B", "", "2099-01-02 08:00:00"),
        )
        s.conn.commit()
        a = s.conn.execute("SELECT id FROM articles WHERE url = ?", ("https://e.com/1",)).fetchone()["id"]
        b = s.conn.execute("SELECT id FROM articles WHERE url = ?", ("https://e.com/2",)).fetchone()["id"]
        s.save_analysis(a, {
            "summary": "股价暴跌", "sentiment": "负面", "confidence": 0.95,
            "tags": [], "entities": [],
        }, tokens=0)
        s.save_related(b, [{"id": a, "title": "某公司股价暴跌", "similarity": 0.9}])
        s.close()
        return tmp_path, out_dir

    def test_webhook_dry_run_prints_payload_without_sending(self, seeded, monkeypatch, capsys):
        _, out_dir = seeded
        monkeypatch.setattr(
            sys, "argv",
            ["run_daily.py", "--no-send", "--day", "2099-12-31",
             "--webhook", "https://oapi.dingtalk.com/robot/send?access_token=test",
             "--webhook-dry-run"],
        )
        with mock.patch("src.alert.requests.post") as post:
            run_daily.main()
        out = capsys.readouterr().out
        assert "[DRY-RUN][告警]" in out
        assert "负面事件告警：某公司股价暴跌" in out
        assert "0.95" in out
        assert "dry-run：1 个负面事件待推送" in out
        post.assert_not_called()  # dry-run 不实际发送
        assert list(out_dir.glob("daily_*.html"))  # 日报仍正常落盘

    def test_real_push_posts_payload(self, seeded, monkeypatch, capsys):
        _, out_dir = seeded
        monkeypatch.setattr(
            sys, "argv",
            ["run_daily.py", "--no-send", "--day", "2099-12-31",
             "--webhook", "https://oapi.dingtalk.com/robot/send?access_token=test"],
        )
        resp = mock.Mock()
        resp.json.return_value = {"errcode": 0}
        with mock.patch("src.alert.requests.post", return_value=resp) as post:
            run_daily.main()
        assert post.call_count == 1
        payload = post.call_args.kwargs["json"]
        assert payload["msgtype"] == "markdown"
        assert "负面事件告警" in payload["markdown"]["text"]
        out = capsys.readouterr().out
        assert "推送成功 1 个" in out
        assert list(out_dir.glob("daily_*.html"))

    def test_no_webhook_skips_silently(self, seeded, monkeypatch, capsys):
        _, out_dir = seeded
        monkeypatch.delenv("ALERT_WEBHOOK", raising=False)
        monkeypatch.setattr(
            sys, "argv", ["run_daily.py", "--no-send", "--day", "2099-12-31"]
        )
        with mock.patch("src.alert.requests.post") as post:
            run_daily.main()
        post.assert_not_called()
        out = capsys.readouterr().out
        assert "未配置 webhook，跳过" in out
        assert list(out_dir.glob("daily_*.html"))

    def test_webhook_idempotency_skips_second_run_unless_force_alert(self, seeded, monkeypatch, capsys):
        _, out_dir = seeded
        argv_base = [
            "run_daily.py", "--no-send", "--day", "2099-12-31",
            "--webhook", "https://oapi.dingtalk.com/robot/send?access_token=test"
        ]
        resp = mock.Mock(status_code=200)
        resp.json.return_value = {"errcode": 0}

        # 首次运行：正常推送并记录
        monkeypatch.setattr(sys, "argv", argv_base)
        with mock.patch("src.alert.requests.post", return_value=resp) as post:
            run_daily.main()
        assert post.call_count == 1
        out1 = capsys.readouterr().out
        assert "推送成功 1 个" in out1

        # 第二次运行：命中去重跳过，不发起网络请求
        monkeypatch.setattr(sys, "argv", argv_base)
        with mock.patch("src.alert.requests.post", return_value=resp) as post:
            run_daily.main()
        assert post.call_count == 0
        out2 = capsys.readouterr().out
        assert "此前已推送过，已自动跳过" in out2

        # 第三次运行：带 --force-alert 强制重新推送
        monkeypatch.setattr(sys, "argv", argv_base + ["--force-alert"])
        with mock.patch("src.alert.requests.post", return_value=resp) as post:
            run_daily.main()
        assert post.call_count == 1
        out3 = capsys.readouterr().out
        assert "推送成功 1 个" in out3



class TestRunDailyIdempotency:
    @pytest.fixture
    def mock_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("ALERT_WEBHOOK", raising=False)
        monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
        monkeypatch.setenv("SMTP_PORT", "465")
        monkeypatch.setenv("SMTP_USER", "a@example.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SMTP_TO", "to@example.com")
        cfg = load_config()
        cfg["storage"]["db_path"] = str(tmp_path / "intel_test.db")
        monkeypatch.setattr(run_daily, "load_config_or_exit", lambda: cfg)
        out_dir = tmp_path / "out"
        monkeypatch.setattr(run_daily, "OUTPUT_DIR", str(out_dir))
        monkeypatch.setattr(
            run_daily, "run_crawl",
            lambda cfg, storage: {"fetched": 0, "matched": 0, "inserted": 0, "errors": []},
        )
        return tmp_path, out_dir

    def test_skips_when_already_sent_unless_force(self, mock_env, monkeypatch, capsys):
        monkeypatch.setattr(
            sys, "argv", ["run_daily.py", "--day", "2026-09-09"]
        )
        with mock.patch("run_daily.send_html") as send_mock:
            # 第一次执行：成功发送
            run_daily.main()
            assert send_mock.call_count == 1
            out = capsys.readouterr().out
            assert "已发送至 to@example.com" in out

            # 第二次执行：拦截跳过
            run_daily.main()
            assert send_mock.call_count == 1  # 依然为 1，未触发发送
            out2 = capsys.readouterr().out
            assert "【跳过】2026-09-09 日报此前已发送过" in out2

        # 第三次执行：带 --force 强制重发
        monkeypatch.setattr(
            sys, "argv", ["run_daily.py", "--day", "2026-09-09", "--force"]
        )
        with mock.patch("run_daily.send_html") as send_mock:
            run_daily.main()
            assert send_mock.call_count == 1
            out3 = capsys.readouterr().out
            assert "已发送至 to@example.com" in out3


class TestRotateHtmlReports:
    def test_rotates_old_html_reports(self, tmp_path):
        out_dir = tmp_path / "reports"
        out_dir.mkdir()
        # 创建 35 个报表文件
        for i in range(1, 36):
            day_str = f"2026-08-{i:02d}"
            (out_dir / f"daily_{day_str}.html").write_text("content", encoding="utf-8")

        # 执行轮转保留最近 30 个
        run_daily._rotate_html_reports(str(out_dir), keep_count=30)

        remaining = sorted([p.name for p in out_dir.glob("daily_*.html")])
        assert len(remaining) == 30
        assert "daily_2026-08-01.html" not in remaining
        assert "daily_2026-08-05.html" not in remaining
        assert "daily_2026-08-06.html" in remaining
        assert "daily_2026-08-35.html" in remaining

    def test_nonexistent_dir_does_not_raise(self, tmp_path):
        run_daily._rotate_html_reports(str(tmp_path / "not_exist"), keep_count=30)


