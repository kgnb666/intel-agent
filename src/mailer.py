"""邮件推送模块：SMTP 发送 HTML 日报。

配置全部走环境变量（密钥不落库不落配置文件）：
    SMTP_HOST     如 smtp.qq.com / smtp.163.com
    SMTP_PORT     通常 465（SSL）或 587（STARTTLS）
    SMTP_USER     发件邮箱
    SMTP_PASSWORD 邮箱授权码（不是登录密码）
    SMTP_TO       收件人，多个用英文逗号分隔
"""
import os
import smtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import formataddr


class MailConfig:
    def __init__(self):
        self.host = os.environ.get("SMTP_HOST", "")
        self.port = int(os.environ.get("SMTP_PORT", "465"))
        self.user = os.environ.get("SMTP_USER", "")
        self.password = os.environ.get("SMTP_PASSWORD", "")
        self.to = [t.strip() for t in os.environ.get("SMTP_TO", "").split(",") if t.strip()]

    @property
    def available(self) -> bool:
        return all([self.host, self.user, self.password, self.to])


def send_html(cfg: MailConfig, subject: str, html_body: str):
    """发送 HTML 邮件。465 走 SSL，其余端口按 STARTTLS 处理。"""
    msg = MIMEText(html_body, "html", "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg["From"] = formataddr((str(Header("情报Agent", "utf-8")), cfg.user))
    msg["To"] = ", ".join(cfg.to)

    if cfg.port == 465:
        server = smtplib.SMTP_SSL(cfg.host, cfg.port, timeout=30)
    else:
        server = smtplib.SMTP(cfg.host, cfg.port, timeout=30)
        server.starttls()
    try:
        server.login(cfg.user, cfg.password)
        server.sendmail(cfg.user, cfg.to, msg.as_string())
    finally:
        server.quit()
