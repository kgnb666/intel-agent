"""配置加载模块：读取 config.yaml，提供全局配置对象。

启动时做字段级校验：不合规的配置在入口立即报错并指出 config.yaml 的出错段，
避免运行到一半才 KeyError / 值类型错误。
"""
import os
import re
import sys

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
_REQUIRED_SECTIONS = ("industry", "sources", "storage", "crawl", "llm", "embedding")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ConfigError(ValueError):
    """配置校验失败。section 记录出错位置（config.yaml 的段名），供入口提示定位。"""

    def __init__(self, message: str, section: str = None):
        super().__init__(message)
        self.section = section


def _fail(section: str, field: str, reason: str):
    raise ConfigError(f"config.yaml 校验失败：{field} {reason}", section=section)


def _check_env_name(section: str, section_cfg: dict, field: str):
    """api_key_env 等字段必须是合法的环境变量名（如 LLM_API_KEY）。"""
    value = section_cfg.get(field)
    if not isinstance(value, str) or not _ENV_NAME_RE.match(value):
        _fail(section, f"{section}.{field}", "必须是合法的环境变量名（字母/数字/下划线，不能以数字开头）")


def _validate(cfg) -> None:
    if not isinstance(cfg, dict) or not cfg:
        raise ConfigError("config.yaml 校验失败：顶层必须是包含六个段的键值映射（当前为空或格式错误）")
    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            _fail(section, section, "段缺失")
        if section == "sources":
            continue  # sources 是数据源列表，类型/内容在下方单独校验
        if not isinstance(cfg[section], dict):
            _fail(section, section, "段不是键值映射")

    industry = cfg["industry"]
    if not isinstance(industry.get("name"), str) or not industry["name"].strip():
        _fail("industry", "industry.name", "不能为空字符串")
    keywords = industry.get("keywords")
    if not isinstance(keywords, list) or not keywords:
        _fail("industry", "industry.keywords", "不能为空列表（至少保留一个关注关键词）")
    if any(not isinstance(k, str) or not k.strip() for k in keywords):
        _fail("industry", "industry.keywords", "每一项必须是非空字符串")

    sources = cfg["sources"]
    if not isinstance(sources, list) or not sources:
        _fail("sources", "sources", "必须是包含至少一个数据源的列表")
    for i, s in enumerate(sources):
        if not isinstance(s, dict) or not isinstance(s.get("url"), str) or not s["url"].strip():
            _fail("sources", f"sources[{i}].url", "必须是非空 URL 字符串")

    db_path = cfg["storage"].get("db_path")
    if not isinstance(db_path, str) or not db_path.strip():
        _fail("storage", "storage.db_path", "不能为空字符串")

    crawl = cfg["crawl"]
    max_items = crawl.get("max_items_per_source")
    if isinstance(max_items, bool) or not isinstance(max_items, int) or max_items <= 0:
        _fail("crawl", "crawl.max_items_per_source", "必须是正整数")
    title_threshold = crawl.get("title_similarity_threshold")
    if isinstance(title_threshold, bool) or not isinstance(title_threshold, (int, float)) or not 0 <= title_threshold <= 1:
        _fail("crawl", "crawl.title_similarity_threshold", "必须是 [0, 1] 区间内的数字")

    _check_env_name("llm", cfg["llm"], "api_key_env")
    _check_env_name("embedding", cfg["embedding"], "api_key_env")
    emb_sim = cfg["embedding"].get("similarity_threshold")
    if isinstance(emb_sim, bool) or not isinstance(emb_sim, (int, float)) or not 0 <= emb_sim <= 1:
        _fail("embedding", "embedding.similarity_threshold", "必须是 [0, 1] 区间内的数字")
    top_k = cfg["embedding"].get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        _fail("embedding", "embedding.top_k", "必须是正整数")

    alert = cfg.get("alert")
    if alert is not None:
        if not isinstance(alert, dict):
            _fail("alert", "alert", "段不是键值映射")
        if alert.get("webhook") is not None and not isinstance(alert["webhook"], str):
            _fail("alert", "alert.webhook", "必须是字符串 URL")
        alert_conf = alert.get("min_confidence")
        if alert_conf is not None and (
            isinstance(alert_conf, bool)
            or not isinstance(alert_conf, (int, float))
            or not 0 <= alert_conf <= 1
        ):
            _fail("alert", "alert.min_confidence", "必须是 [0, 1] 区间内的数字")


def load_config(path: str = _CONFIG_PATH) -> dict:
    """读取并校验配置。任何不合规字段都会抛出带字段名/段名的 ConfigError。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"找不到配置文件：{path}", section="文件")
    except OSError as e:
        raise ConfigError(f"无法读取配置文件 {path}：{e}", section="文件")
    except yaml.YAMLError as e:
        raise ConfigError(f"config.yaml 解析失败（YAML 语法错误）：{e}", section="文件")

    _validate(cfg)

    # db_path 转为绝对路径，保证在任何工作目录下运行都定位到同一个库
    base = os.path.dirname(path)
    cfg["storage"]["db_path"] = os.path.normpath(os.path.join(base, cfg["storage"]["db_path"]))
    # 环境变量可覆盖库路径：部署/演示时指向快照库（如 data/intel.demo.db），不动线上数据
    if os.environ.get("INTEL_DB_PATH"):
        cfg["storage"]["db_path"] = os.path.normpath(
            os.path.join(base, os.environ["INTEL_DB_PATH"])
        )
    return cfg


def load_config_or_exit(path: str = _CONFIG_PATH) -> dict:
    """入口脚本专用：配置错误时打印友好提示并以退出码 1 结束，而不是抛 traceback。"""
    try:
        return load_config(path)
    except ConfigError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        if e.section and e.section != "文件":
            print(f"请检查 config.yaml 的「{e.section}」段后重试。", file=sys.stderr)
        else:
            print("请检查 config.yaml 文件本身（路径、格式、编码）后重试。", file=sys.stderr)
        sys.exit(1)


def save_keywords(keywords: list, path: str = _CONFIG_PATH) -> None:
    """更新 config.yaml 中的关键词列表，保留原有全部注释、结构与相对路径。"""
    if not isinstance(keywords, list) or not keywords:
        _fail("industry", "industry.keywords", "不能为空列表（至少保留一个关注关键词）")
    clean_kws = [str(k).strip() for k in keywords if str(k).strip()]
    if not clean_kws:
        _fail("industry", "industry.keywords", "每一项必须是非空字符串")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        raise ConfigError(f"找不到配置文件：{path}", section="文件")
    except OSError as e:
        raise ConfigError(f"无法读取配置文件 {path}：{e}", section="文件")

    # 匹配 keywords: 及其下属所有以 '-' 开头的列表项行
    pattern = re.compile(
        r"(^[ \t]*keywords:\s*\r?\n)(?:[ \t]*-[ \t]+.*(?:\r?\n|$))+",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if not match:
        raise ConfigError("在配置文件中未找到 keywords 列表段落", section="industry")

    nl = "\r\n" if "\r\n" in match.group(1) else "\n"
    kw_lines = []
    for k in clean_kws:
        if any(ch in k for ch in ":#{}[]&*?|<>=!%@`,'\" \t"):
            item_str = yaml.safe_dump([k], allow_unicode=True).strip()
        else:
            item_str = f"- {k}"
        kw_lines.append(f"    {item_str}{nl}")

    replacement = match.group(1) + "".join(kw_lines)
    new_content = content[:match.start()] + replacement + content[match.end():]

    # 写入前先校验合成后的配置语法与结构完整性
    try:
        test_cfg = yaml.safe_load(new_content)
    except yaml.YAMLError as e:
        raise ConfigError(f"更新后 config.yaml 语法错误：{e}", section="industry")

    _validate(test_cfg)

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        raise ConfigError(f"无法写入配置文件 {path}：{e}", section="文件")


if __name__ == "__main__":
    c = load_config_or_exit()
    print("行业:", c["industry"]["name"])
    print("数据源:", [s["name"] for s in c["sources"]])
    print("数据库:", c["storage"]["db_path"])
