"""logger 模块单元测试：测试 logger 获取、单例 handlers 防重、格式规范及文件轮转属性。"""
import logging
from logging.handlers import RotatingFileHandler

from src.logger import DynamicStdoutHandler, get_logger


class TestLogger:
    def test_get_logger_basic(self):
        log = get_logger("test_logger_basic")
        assert log.level == logging.INFO
        assert len(log.handlers) >= 2

        # 验证包含 DynamicStdoutHandler 与 RotatingFileHandler
        has_stdout = any(isinstance(h, DynamicStdoutHandler) for h in log.handlers)
        has_file = any(isinstance(h, RotatingFileHandler) for h in log.handlers)
        assert has_stdout is True
        assert has_file is True

    def test_get_logger_idempotent_no_duplicate_handlers(self):
        log1 = get_logger("test_duplicate")
        count_first = len(log1.handlers)
        log2 = get_logger("test_duplicate")
        count_second = len(log2.handlers)
        assert count_first == count_second
        assert log1 is log2

    def test_file_handler_properties(self):
        log = get_logger("test_rfh_props")
        rfh = next(h for h in log.handlers if isinstance(h, RotatingFileHandler))
        assert rfh.maxBytes == 5 * 1024 * 1024
        assert rfh.backupCount == 3
        assert rfh.encoding.lower() == "utf-8"

    def test_log_formatter_pattern(self):
        log = get_logger("test_formatter")
        formatter = log.handlers[0].formatter
        assert formatter is not None
        record = logging.LogRecord(
            name="test_formatter",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="测试日志消息",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "[INFO]" in formatted
        assert "[test_formatter]" in formatted
        assert "测试日志消息" in formatted
