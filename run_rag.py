"""入口：执行 RAG 流程（向量化 -> 相似事件关联 -> 周度趋势提炼）。

用法：
    python run_rag.py            # 正常运行（需配置 EMBEDDING_API_KEY 或 LLM_API_KEY）
    python run_rag.py --dry-run  # 只打印将做的事，不调用 API
"""
import argparse
import json

from src.analyzer import LLMConfig
from src.config import load_config
from src.rag import RAG, EmbeddingConfig
from src.storage import Storage


def main():
    parser = argparse.ArgumentParser(description="RAG 历史情报关联")
    parser.add_argument("--dry-run", action="store_true", help="只打印流程，不实际调用 API")
    args = parser.parse_args()

    cfg = load_config()
    emb_cfg = EmbeddingConfig(cfg)
    llm_cfg = LLMConfig(cfg)

    if not args.dry_run and not emb_cfg.available:
        print("未检测到 embedding API key（环境变量 EMBEDDING_API_KEY 或 LLM_API_KEY）。")
        print("请设置后重试，或使用 --dry-run 模式验证流程。")
        return

    storage = Storage(cfg["storage"]["db_path"])
    try:
        rag = RAG(emb_cfg, cfg["industry"]["name"], llm_cfg=llm_cfg, dry_run=args.dry_run)
        stats = rag.run(storage, dry_run=args.dry_run)

        if not args.dry_run:
            # 输出关联示例，直观验证效果
            rows = storage.conn.execute(
                """SELECT id, title, related_event FROM articles
                   WHERE related_event IS NOT NULL AND related_event != '[]'
                   ORDER BY id DESC LIMIT 5"""
            ).fetchall()
            if rows:
                print("\n关联示例（最近 5 条有相关事件的情报）:")
                for r in rows:
                    print(f"  《{r['title']}》")
                    for rel in json.loads(r["related_event"]):
                        print(f"    └─ 关联 《{rel['title']}》 (相似度 {rel['similarity']})")
            trends = storage.latest_trends(3)
            if trends:
                print("\n最新趋势脉络:")
                for t in trends:
                    print(f"  [{t['week_start']}] {t['content']}")
    finally:
        storage.close()

    print(f"\n向量化 {stats['vectorized']} 篇 | 新关联 {stats['related']} 篇 | 生成趋势 {stats['trends']} 条")


if __name__ == "__main__":
    main()
