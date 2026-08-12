"""配置加载模块：读取 config.yaml，提供全局配置对象。"""
import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


def load_config(path: str = _CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # db_path 转为绝对路径，保证在任何工作目录下运行都定位到同一个库
    base = os.path.dirname(path)
    cfg["storage"]["db_path"] = os.path.normpath(os.path.join(base, cfg["storage"]["db_path"]))
    # 环境变量可覆盖库路径：部署/演示时指向快照库（如 data/intel.demo.db），不动线上数据
    if os.environ.get("INTEL_DB_PATH"):
        cfg["storage"]["db_path"] = os.path.normpath(
            os.path.join(base, os.environ["INTEL_DB_PATH"])
        )
    return cfg


if __name__ == "__main__":
    c = load_config()
    print("行业:", c["industry"]["name"])
    print("数据源:", [s["name"] for s in c["sources"]])
    print("数据库:", c["storage"]["db_path"])
