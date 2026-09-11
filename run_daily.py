"""入口：一键执行完整流水线（采集 → 分析 → RAG 关联 → 生成报告 → 发邮件）。

用法：
    python run_daily.py            # 完整跑一遍并发送邮件
    python run_daily.py --no-send  # 只生成 HTML 报告到 output/ 目录，不发邮件
    python run_daily.py --webhook https://oapi.dingtalk.com/robot/send?access_token=xxx
                                   # 负面高置信度事件推送钉钉/企业微信 webhook
    python run_daily.py --webhook <url> --webhook-dry-run  # 只打印告警 JSON，不发送

设计：每一步独立 try/except，某步失败不中断后续步骤，
失败信息记入降级说明，最终体现在邮件/报告里——半成品日报好过没有日报。
"""
import argparse
import json
import os
from datetime import date, timedelta

from src.alert import AlertConfig, alert_candidates, build_payload, filter_unsent_alerts, push_alerts
from src.analyzer import Analyzer, LLMConfig
from src.config import load_config_or_exit
from src.crawler import run_crawl
from src.events import aggregate_events
from src.logger import get_logger
from src.mailer import MailConfig, send_html
from src.rag import RAG, EmbeddingConfig
from src.report import collect_daily, render_html, render_plain_text
from src.storage import Storage

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
logger = get_logger("run_daily")


def _rotate_html_reports(output_dir: str, keep_count: int = 30):
    """保留 output_dir 下最近 keep_count 个 daily_*.html 报表，多余的安全清理。"""
    if not os.path.exists(output_dir):
        return
    files = [
        f for f in os.listdir(output_dir)
        if f.startswith("daily_") and f.endswith(".html")
    ]
    files.sort()
    if len(files) > keep_count:
        to_delete = files[:-keep_count]
        for f in to_delete:
            try:
                os.remove(os.path.join(output_dir, f))
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="每日情报流水线")
    parser.add_argument("--no-send", action="store_true", help="只生成 HTML 报告文件，不发邮件")
    parser.add_argument("--day", default=None, help="报告日期 YYYY-MM-DD，默认昨天")
    parser.add_argument("--webhook", default=None, help="钉钉/企业微信 webhook 地址（覆盖 config.yaml alert.webhook）")
    parser.add_argument("--webhook-dry-run", action="store_true", help="只打印告警 JSON，不实际推送")
    parser.add_argument("--force", action="store_true", help="强制重新发送邮件（忽略已发送检查）")
    parser.add_argument("--force-alert", action="store_true", help="强制重新推送告警（忽略已推送历史检查）")
    parser.add_argument("--prune-days", type=int, default=90, help="保留最近 N 天数据，传 0 表示不清理（默认 90 天）")
    args = parser.parse_args()

    cfg = load_config_or_exit()
    storage = Storage(cfg["storage"]["db_path"])
    degraded = []

    try:
        # 1. 采集
        try:
            stats = run_crawl(cfg, storage)
            logger.info(f"[采集] 抓取 {stats['fetched']} | 命中 {stats['matched']} | 新入库 {stats['inserted']}")
            if stats["errors"]:
                degraded.append("部分数据源失败：" + "；".join(stats["errors"]))
                logger.warning(f"[采集] 部分数据源失败：{'；'.join(stats['errors'])}")
        except Exception as e:
            degraded.append(f"采集环节失败：{e}")
            logger.error(f"[采集] 失败：{e}")

        # 2. LLM 分析
        llm_cfg = LLMConfig(cfg)
        if llm_cfg.available:
            try:
                st = Analyzer(llm_cfg, cfg["industry"]["name"]).run(storage)
                logger.info(f"[分析] 处理 {st['total']} | 成功 {st['success']} | 失败 {st['failed']} | token {st['tokens']}")
                if st["failed"]:
                    degraded.append(f"{st['failed']} 篇文章分析失败")
                    logger.warning(f"[分析] {st['failed']} 篇文章分析失败")
            except Exception as e:
                degraded.append(f"分析环节失败：{e}")
                logger.error(f"[分析] 失败：{e}")
        else:
            degraded.append("未配置 LLM API key，跳过智能分析")
            logger.info("[分析] 未配置 key，跳过")

        # 3. RAG 关联
        emb_cfg = EmbeddingConfig(cfg)
        if emb_cfg.available:
            try:
                st = RAG(emb_cfg, cfg["industry"]["name"], llm_cfg=llm_cfg).run(storage)
                logger.info(f"[RAG] 向量化 {st['vectorized']} | 关联 {st['related']} | 趋势 {st['trends']}")
            except Exception as e:
                degraded.append(f"RAG 环节失败：{e}")
                logger.error(f"[RAG] 失败：{e}")
        else:
            logger.info("[RAG] 未配置 embedding key，跳过")

        # 3.5 事件聚合 + 负面高置信度实时告警
        alert_cfg = AlertConfig(cfg, webhook=args.webhook)
        if alert_cfg.available:
            try:
                events = aggregate_events(storage, days=2)
                alerts = alert_candidates(events, alert_cfg.min_confidence)
                total_candidates = len(alerts)
                if not args.force_alert:
                    alerts = filter_unsent_alerts(alerts, storage)
                    skipped = total_candidates - len(alerts)
                    if skipped > 0:
                        logger.info(f"[告警] 命中 {total_candidates} 个负面事件，其中 {skipped} 个此前已推送过，已自动跳过（使用 --force-alert 可强制重发）")
                if args.webhook_dry_run:
                    for ev in alerts:
                        payload = build_payload(alert_cfg.webhook, ev, cfg["industry"]["name"])
                        logger.info(f"[DRY-RUN][告警] {json.dumps(payload, ensure_ascii=False)}")
                    logger.info(f"[告警] dry-run：{len(alerts)} 个负面事件待推送（未实际发送）")
                else:
                    sent = push_alerts(alert_cfg.webhook, alerts, cfg["industry"]["name"], storage=storage)
                    logger.info(f"[告警] 负面事件待推送 {len(alerts)} 个 | 推送成功 {sent} 个")
                    if sent < len(alerts):
                        degraded.append(f"{len(alerts) - sent} 个告警推送失败")
                        logger.warning(f"[告警] {len(alerts) - sent} 个告警推送失败")
            except Exception as e:
                degraded.append(f"告警环节失败：{e}")
                logger.error(f"[告警] 失败：{e}")
        else:
            logger.info("[告警] 未配置 webhook，跳过")

        # 4. 生成报告
        day = args.day or (date.today() - timedelta(days=1)).isoformat()
        data = collect_daily(storage, day=day)
        # 若默认昨天没数据（比如今天才部署），回退到今天，避免发出空日报
        if not args.day and data["total"] == 0:
            day = date.today().isoformat()
            data = collect_daily(storage, day=day)
        html_body = render_html(data, cfg["industry"]["name"], degraded_notes=degraded)
        plain_body = render_plain_text(data, cfg["industry"]["name"], degraded_notes=degraded)
        logger.info(f"[报告] {day} 日报已生成（{data['total']} 条情报）")

        # 5. 发送 / 落盘
        mail_cfg = MailConfig()
        if args.no_send or not mail_cfg.available:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f"daily_{day}.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_body)
            logger.info(f"[输出] 报告已保存到 {out_path}" + ("" if args.no_send else "（未配置 SMTP，未发送）"))
        else:
            if storage.has_report_sent(day) and not args.force:
                logger.info(f"[邮件] 【跳过】{day} 日报此前已发送过，跳过（使用 --force 可强制重发）")
            else:
                send_html(mail_cfg, f"【{cfg['industry']['name']}】情报日报 {day}", html_body, plain_body=plain_body)
                storage.record_report_sent(day)
                logger.info(f"[邮件] 已发送至 {', '.join(mail_cfg.to)}")

        # 6. 数据生命周期修剪与历史报表轮转
        if args.prune_days > 0 and hasattr(storage, "prune_history"):
            try:
                prune_res = storage.prune_history(args.prune_days)
                del_n = prune_res.get("deleted_articles", 0)
                if del_n > 0:
                    logger.info(f"[清理] 已淘汰 {del_n} 篇超期文章（保留最近 {args.prune_days} 天）")
            except Exception as e:
                logger.warning(f"[清理] 数据修剪异常：{e}")

        try:
            _rotate_html_reports(OUTPUT_DIR, keep_count=30)
        except Exception as e:
            logger.warning(f"[清理] 报表轮转异常：{e}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
