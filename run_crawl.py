"""入口：执行一轮情报采集。"""
from src.config import load_config
from src.storage import Storage
from src.crawler import run_crawl


def main():
    cfg = load_config()
    storage = Storage(cfg["storage"]["db_path"])
    try:
        stats = run_crawl(cfg, storage)
    finally:
        total = storage.count()
        storage.close()

    print(f"行业主题: {cfg['industry']['name']}")
    print(f"抓取 {stats['fetched']} 条 | 命中关键词 {stats['matched']} 条 | 新入库 {stats['inserted']} 条")
    print(f"库内累计: {total} 条")
    if stats["errors"]:
        print("失败源:")
        for e in stats["errors"]:
            print(" -", e)


if __name__ == "__main__":
    main()
