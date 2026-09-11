"""统一日志体系：控制台输出 + 自动轮转落盘到 output/agent.log。"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_FILE = os.path.join(LOG_DIR, "agent.log")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


class DynamicStdoutHandler(logging.StreamHandler):
    """使用动态 sys.stdout 的 StreamHandler，确保与 pytest capsys 等重定向无缝兼容。"""

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def get_logger(name: str = "intel-agent") -> logging.Logger:
    """获取标准 Logger，同时向控制台 (sys.stdout) 与 output/agent.log 写入。

    单文件最大 5MB，最多保留 3 个历史备份。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)

        # 1. 控制台输出处理器
        console_handler = DynamicStdoutHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. 轮转文件处理器
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False

    return logger
