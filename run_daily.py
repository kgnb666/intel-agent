"""入口：一键执行完整流水线（采集 → 分析 → RAG 关联 → 生成报告 → 发邮件）。

用法：
    python run_daily.py            # 完整跑一遍并发送邮件
    python run_daily.py --no-send  # 只生成 HTML 报告到 output/ 目录，不发邮件

设计：每一步独立 try/except，某步失败不中断后续步骤，
失败信息记入降级说明，最终体现在邮件/报告里——半成品日报好过没有日报。
"""
import argparse
import os
from datetime import date, timedelta

from src.analyzer import Analyzer, LLMConfig
from src.config import load_config
from src.crawler import run_crawl
from src.mailer import MailConfig, send_html
from src.rag import RAG, EmbeddingConfig
from src.report import collect_daily, render_html
from src.storage import Storage

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def log(step: str, msg: str):
    print(f"[{step}] {msg}")


def main():
    parser = argparse.ArgumentParser(description="每日情报流水线")
    parser.add_argument("--no-send", action="store_true", help="只生成 HTML 报告文件，不发邮件")
    parser.add_argument("--day", default=None, help="报告日期 YYYY-MM-DD，默认昨天")
    args = parser.parse_args()

    cfg = load_config()
    storage = Storage(cfg["storage"]["db_path"])
    degraded = []

    try:
        # 1. 采集
        try:
            stats = run_crawl(cfg, storage)
            log("采集", f"抓取 {stats['fetched']} | 命中 {stats['matched']} | 新入库 {stats['inserted']}")
            if stats["errors"]:
                degraded.append("部分数据源失败：" + "；".join(stats["errors"]))
        except Exception as e:
            degraded.append(f"采集环节失败：{e}")
            log("采集", f"失败：{e}")

        # 2. LLM 分析
        llm_cfg = LLMConfig(cfg)
        if llm_cfg.available:
            try:
                st = Analyzer(llm_cfg, cfg["industry"]["name"]).run(storage)
                log("分析", f"处理 {st['total']} | 成功 {st['success']} | 失败 {st['failed']} | token {st['tokens']}")
                if st["failed"]:
                    degraded.append(f"{st['failed']} 篇文章分析失败")
            except Exception as e:
                degraded.append(f"分析环节失败：{e}")
                log("分析", f"失败：{e}")
        else:
            degraded.append("未配置 LLM API key，跳过智能分析")
            log("分析", "未配置 key，跳过")

        # 3. RAG 关联
        emb_cfg = EmbeddingConfig(cfg)
        if emb_cfg.available:
            try:
                st = RAG(emb_cfg, cfg["industry"]["name"], llm_cfg=llm_cfg).run(storage)
                log("RAG", f"向量化 {st['vectorized']} | 关联 {st['related']} | 趋势 {st['trends']}")
            except Exception as e:
                degraded.append(f"RAG 环节失败：{e}")
                log("RAG", f"失败：{e}")
        else:
            log("RAG", "未配置 embedding key，跳过")

        # 4. 生成报告
        day = args.day or (date.today() - timedelta(days=1)).isoformat()
        data = collect_daily(storage, day=day)
        # 若昨天没数据（比如今天才部署），回退到今天，避免发出空日报
        if data["total"] == 0:
            day = date.today().isoformat()
            data = collect_daily(storage, day=day)
        html_body = render_html(data, cfg["industry"]["name"], degraded_notes=degraded)
        log("报告", f"{day} 日报已生成（{data['total']} 条情报）")

        # 5. 发送 / 落盘
        mail_cfg = MailConfig()
        if args.no_send or not mail_cfg.available:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_path = os.path.join(OUTPUT_DIR, f"daily_{day}.html")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(html_body)
            log("输出", f"报告已保存到 {out_path}" + ("" if args.no_send else "（未配置 SMTP，未发送）"))
        else:
            send_html(mail_cfg, f"【{cfg['industry']['name']}】情报日报 {day}", html_body)
            log("邮件", f"已发送至 {', '.join(mail_cfg.to)}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
