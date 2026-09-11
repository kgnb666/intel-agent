"""告警模块：负面高置信度事件通过 Webhook 推送（钉钉 / 企业微信兼容）。

规则：sentiment=负面 且 置信度 >= min_confidence 的事件才告警。
无 webhook 配置时静默跳过；推送失败只记日志、不中断流水线（降级文化）。
"""
import logging
import os

import requests

logger = logging.getLogger("alert")


class AlertConfig:
    """告警配置。优先级：命令行 --webhook > 环境变量 ALERT_WEBHOOK > config.yaml alert.webhook。"""

    def __init__(self, cfg: dict, webhook: str = None):
        alert = cfg.get("alert", {}) or {}
        self.webhook = (
            webhook
            or os.environ.get("ALERT_WEBHOOK", "")
            or alert.get("webhook", "")
            or ""
        )
        self.min_confidence = float(alert.get("min_confidence", 0.8))

    @property
    def available(self) -> bool:
        return bool(self.webhook)


def alert_candidates(events: list, min_confidence: float = 0.8) -> list:
    """筛出包含负面文章、且最高负面置信度 >= min_confidence 的事件。"""
    result = []
    for ev in events:
        confs = [
            a.get("sentiment_conf") or 0
            for a in ev.get("articles", [])
            if a.get("sentiment") == "负面"
        ]
        if confs and max(confs) >= min_confidence:
            result.append(ev)
    return result


def _get_event_core_article_id(ev: dict) -> int:
    """提取事件的核心报道 ID（最高负面置信度文章，若无则取 min_id/id）。"""
    event_id = ev.get("id", 0)
    neg_articles = [
        a for a in ev.get("articles", [])
        if a.get("sentiment") == "负面"
    ]
    if neg_articles:
        best = max(neg_articles, key=lambda a: a.get("sentiment_conf") or 0)
        return best.get("id", ev.get("min_id", event_id))
    return ev.get("min_id", event_id)


def filter_unsent_alerts(alerts: list, storage) -> list:
    """过滤掉已发送过告警的事件。
    以事件的 id 与核心文章 article_id 作为去重键。
    """
    if storage is None:
        return alerts
    unsent = []
    for ev in alerts:
        event_id = ev.get("id", 0)
        core_art_id = _get_event_core_article_id(ev)
        if not storage.is_alert_sent(event_id, core_art_id):
            unsent.append(ev)
    return unsent


def build_payload(webhook: str, event: dict, industry: str = "") -> dict:
    """构造钉钉/企业微信兼容的 markdown 消息 JSON。

    两者协议差异只在字段名：企业微信用 markdown.content，钉钉用 markdown.text。
    按 webhook 域名自动选择；其他兼容网关统一走钉钉格式。
    """
    neg = [
        a.get("sentiment_conf") or 0
        for a in event.get("articles", [])
        if a.get("sentiment") == "负面"
    ]
    top_conf = max(neg) if neg else 0.0
    title = f"⚠️ 负面事件告警：{event['title']}"
    lines = [
        f"### {title}",
        f"- 事件规模：{event['size']} 篇相关报道",
        f"- 负面置信度：{top_conf:.2f}",
        f"- 涉及行业：{industry or '—'}",
        "",
        "**时间线**：",
    ]
    lines += [
        f"- `{t['date']}` [{t['source']}]《{t['title']}》"
        for t in event.get("timeline", [])[:10]
    ]
    text = "\n".join(lines)
    if "qyapi.weixin" in (webhook or ""):
        return {"msgtype": "markdown", "markdown": {"content": text}}
    return {"msgtype": "markdown", "markdown": {"title": title, "text": text}}


def push_alert(webhook: str, payload: dict, timeout: int = 10) -> bool:
    """推送一条告警 JSON。网络错误 / 非 2xx / errcode 非 0 均返回 False 并记录详细排障日志。"""
    try:
        resp = requests.post(webhook, json=payload, timeout=timeout)
        resp.raise_for_status()
        status_code = getattr(resp, "status_code", 200)
        if isinstance(status_code, int) and status_code != 200:
            text = str(getattr(resp, "text", ""))[:200]
            logger.warning(f"Webhook 推送 HTTP 异常 (HTTP {status_code})：{text}")
            return False
        try:
            body = resp.json()
        except ValueError:
            return True  # 非 JSON 网关（如 IFTTT），2xx 即成功
        errcode = body.get("errcode", 0)
        if errcode != 0:
            errmsg = body.get("errmsg", "")
            text = str(getattr(resp, "text", ""))[:200]
            logger.warning(
                f"Webhook 推送被拒 (errcode={errcode}, errmsg={errmsg})：{text}"
            )
            return False
        return True
    except requests.HTTPError as e:
        status = getattr(getattr(e, "response", None), "status_code", "5xx")
        text = str(getattr(getattr(e, "response", None), "text", ""))[:200]
        logger.warning(f"Webhook 推送 HTTP 异常 (HTTP {status})：{text}")
        return False
    except requests.RequestException as e:
        logger.warning(f"Webhook 网络请求失败：{e}")
        return False


def push_alerts(
    webhook: str, events: list, industry: str = "", timeout: int = 10, storage=None
) -> int:
    """逐个推送事件告警，返回成功数。单个失败不影响其余。
    若传入 storage，推送成功后自动记录 sent_alerts 状态。
    """
    sent = 0
    for ev in events:
        payload = build_payload(webhook, ev, industry)
        if push_alert(webhook, payload, timeout=timeout):
            sent += 1
            if storage is not None:
                event_id = ev.get("id", 0)
                core_art_id = _get_event_core_article_id(ev)
                storage.record_alert_sent(event_id, core_art_id)
    return sent
