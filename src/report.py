"""日报生成模块：汇总当日情报，产出 HTML 邮件正文。

HTML 邮件的兼容性约束比网页严格得多：多数邮件客户端（尤其是 QQ 邮箱、
Outlook）不支持 flex/grid 和外部 CSS，因此布局用 table、样式全部内联。
"""
import html
from datetime import date, datetime, timedelta

SENTIMENT_COLOR = {"正面": "#2e9e5b", "中性": "#8a8f98", "负面": "#d64545"}


def collect_daily(storage, day: str = None) -> dict:
    """汇总指定日期（默认昨天，日报通常是"次日发前一日"）的数据。

    重要事件排序依据：情感置信度高的非中性事件优先——情绪波动大的新闻
    往往更重要（暴涨/暴跌/暴雷），中性资讯靠后。
    """
    day = day or (date.today() - timedelta(days=1)).isoformat()
    rows = storage.conn.execute(
        """SELECT a.id, a.title, a.url, a.source, an.summary_ai, an.sentiment,
                  an.sentiment_conf, an.tags
           FROM articles a LEFT JOIN analysis an ON an.article_id = a.id
           WHERE date(a.fetched_at) = ?""",
        (day,),
    ).fetchall()
    articles = [dict(r) for r in rows]

    analyzed = [a for a in articles if a["sentiment"]]
    sentiment_dist = {"正面": 0, "中性": 0, "负面": 0}
    for a in analyzed:
        sentiment_dist[a["sentiment"]] = sentiment_dist.get(a["sentiment"], 0) + 1

    def importance(a):
        base = a["sentiment_conf"] or 0
        return base if a["sentiment"] != "中性" else base * 0.3

    top5 = sorted(analyzed, key=importance, reverse=True)[:5]
    trends = [
        dict(t) for t in storage.conn.execute(
            "SELECT week_start, content FROM trends ORDER BY id DESC LIMIT 1"
        ).fetchall()
    ]
    return {
        "day": day,
        "total": len(articles),
        "analyzed": len(analyzed),
        "sentiment_dist": sentiment_dist,
        "top_events": top5,
        "trends": trends,
        "all": articles,
    }


def _esc(text) -> str:
    return html.escape(str(text or ""))


def render_plain_text(data: dict, industry: str, degraded_notes: list = None) -> str:
    """渲染纯文本邮件正文（text/plain）。作为 HTML 的备用正文，适配终端及纯文本邮件客户端。"""
    d = data
    dist = d.get("sentiment_dist", {})
    dist_text = f"正面 {dist.get('正面', 0)} · 中性 {dist.get('中性', 0)} · 负面 {dist.get('负面', 0)}"

    lines = [
        f"【{industry} · 每日情报日报】",
        f"{d.get('day', '')} · 新增 {d.get('total', 0)} 条 · 已分析 {d.get('analyzed', 0)} 条 · {dist_text}",
        "=" * 50,
    ]

    if degraded_notes:
        lines.append("⚠️ 本次流水线部分环节降级：")
        for n in degraded_notes:
            lines.append(f"  - {n}")
        lines.append("-" * 50)

    lines.append("🔥 今日 Top 事件：")
    top_events = d.get("top_events", [])
    if top_events:
        for i, e in enumerate(top_events, 1):
            title = e.get("title", "")
            sentiment = e.get("sentiment", "中性")
            summary = e.get("summary_ai", "")
            source = e.get("source", "")
            url = e.get("url", "")
            lines.append(f"{i}. [{sentiment}] {title}")
            if summary:
                lines.append(f"   摘要: {summary}")
            info = []
            if source:
                info.append(f"来源: {source}")
            if url:
                info.append(f"链接: {url}")
            if info:
                lines.append(f"   {' | '.join(info)}")
            lines.append("")
    else:
        lines.append("当日无已分析事件\n")

    trends = d.get("trends", [])
    if trends:
        lines.append("-" * 50)
        lines.append("📈 本周趋势脉络：")
        for t in trends:
            lines.append(f"[{t.get('week_start', '')}] {t.get('content', '')}")
        lines.append("")

    lines.append("-" * 50)
    lines.append(f"由「智能行业情报分析 Agent」自动生成 · {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    return "\n".join(lines).strip()


def render_html(data: dict, industry: str, degraded_notes: list = None) -> str:
    """渲染 HTML 邮件正文。table 布局 + 内联样式，兼容主流邮件客户端。"""
    d = data
    dist = d["sentiment_dist"]
    dist_text = f"正面 {dist.get('正面', 0)} · 中性 {dist.get('中性', 0)} · 负面 {dist.get('负面', 0)}"

    event_rows = ""
    for i, e in enumerate(d["top_events"], 1):
        color = SENTIMENT_COLOR.get(e["sentiment"], "#8a8f98")
        event_rows += f"""
        <tr><td style="padding:12px 0;border-bottom:1px solid #eee">
          <div style="font-size:15px;font-weight:bold">
            {i}. <a href="{_esc(e['url'])}" style="color:#1f6feb;text-decoration:none">{_esc(e['title'])}</a>
            <span style="background:{color};color:#fff;font-size:12px;padding:1px 8px;border-radius:10px">{_esc(e['sentiment'])}</span>
          </div>
          <div style="color:#555;font-size:13px;margin-top:6px">{_esc(e['summary_ai'])}</div>
          <div style="color:#999;font-size:12px;margin-top:4px">{_esc(e['source'])}</div>
        </td></tr>"""

    trend_block = ""
    if d["trends"]:
        t = d["trends"][0]
        trend_block = f"""
        <tr><td style="padding:20px 32px">
          <div style="font-size:16px;font-weight:bold;margin-bottom:10px">📈 本周趋势脉络</div>
          <div style="color:#333;font-size:14px;line-height:1.8;background:#f6f8fa;padding:14px;border-radius:6px">{_esc(t['content'])}</div>
        </td></tr>"""

    degraded_block = ""
    if degraded_notes:
        items = "".join(f"<li>{_esc(n)}</li>" for n in degraded_notes)
        degraded_block = f"""
        <tr><td style="padding:12px 32px;color:#b45309;font-size:13px">
          ⚠️ 本次流水线部分环节降级：<ul style="margin:6px 0">{items}</ul>
        </td></tr>"""

    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#f0f2f5">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#f0f2f5;padding:24px 0">
<tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:8px;overflow:hidden;font-family:'Microsoft YaHei',sans-serif">
  <tr><td style="background:#1f6feb;color:#fff;padding:20px 32px">
    <div style="font-size:20px;font-weight:bold">{_esc(industry)} · 每日情报日报</div>
    <div style="font-size:13px;opacity:.85;margin-top:4px">{d['day']} · 新增 {d['total']} 条 · 已分析 {d['analyzed']} 条 · {dist_text}</div>
  </td></tr>
  {degraded_block}
  <tr><td style="padding:20px 32px">
    <div style="font-size:16px;font-weight:bold;margin-bottom:4px">🔥 今日 Top 事件</div>
    <table width="100%" cellpadding="0" cellspacing="0">{event_rows or '<tr><td style="color:#999;padding:12px 0">当日无已分析事件</td></tr>'}</table>
  </td></tr>
  {trend_block}
  <tr><td style="padding:16px 32px;color:#999;font-size:12px;border-top:1px solid #eee">
    由「智能行业情报分析 Agent」自动生成 · {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </td></tr>
</table>
</td></tr></table>
</body></html>"""
