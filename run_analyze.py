"""入口：对未分析文章执行 LLM 分析。

用法：
    python run_analyze.py            # 正常分析（需配置 LLM_API_KEY）
    python run_analyze.py --dry-run  # 只打印 prompt 不调用 API，用于无 key 验证逻辑
    python run_analyze.py --limit 5  # 本次最多处理 5 篇（不超过 config 的 max_per_run）
"""
import argparse

from src.analyzer import Analyzer, LLMConfig
from src.config import load_config
from src.storage import Storage


def main():
    parser = argparse.ArgumentParser(description="LLM 情报分析")
    parser.add_argument("--dry-run", action="store_true", help="只打印 prompt，不实际调用 API")
    parser.add_argument("--limit", type=int, default=None, help="本次最大处理篇数")
    args = parser.parse_args()

    cfg = load_config()
    llm_cfg = LLMConfig(cfg)

    if not args.dry_run and not llm_cfg.available:
        print("未检测到 API key（环境变量 %s）。" % cfg["llm"].get("api_key_env", "LLM_API_KEY"))
        print("请设置后重试，或使用 --dry-run 模式验证流程。")
        return

    storage = Storage(cfg["storage"]["db_path"])
    try:
        analyzer = Analyzer(llm_cfg, cfg["industry"]["name"], dry_run=args.dry_run)
        stats = analyzer.run(storage, limit=args.limit, dry_run=args.dry_run)
    finally:
        storage.close()

    print(f"模型: {llm_cfg.model} @ {llm_cfg.base_url}")
    print(f"处理 {stats['total']} 篇 | 成功 {stats['success']} | 失败 {stats['failed']} | 消耗 token {stats['tokens']}")
    if stats["errors"]:
        print("失败明细:")
        for e in stats["errors"]:
            print(" -", e)


if __name__ == "__main__":
    main()
