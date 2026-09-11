from src.config import load_config_or_exit
from src.crawler import run_crawl
from src.logger import get_logger
from src.storage import Storage

logger = get_logger("run_crawl")


def main():
    cfg = load_config_or_exit()
    storage = Storage(cfg["storage"]["db_path"])
    try:
        stats = run_crawl(cfg, storage)
    finally:
        total = storage.count()
        storage.close()

    logger.info(f"行业主题: {cfg['industry']['name']}")
    logger.info(f"抓取 {stats['fetched']} 条 | 命中关键词 {stats['matched']} 条 | 新入库 {stats['inserted']} 条")
    logger.info(f"库内累计: {total} 条")
    if stats["errors"]:
        logger.warning("失败源:")
        for e in stats["errors"]:
            logger.warning(f" - {e}")


if __name__ == "__main__":
    main()
