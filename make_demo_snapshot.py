"""生成演示数据快照：data/intel.db -> data/demo.db。

用途：Streamlit Cloud 部署/面试演示时，看板需要有分析数据才好看。
真实环境应跑 run_analyze.py / run_rag.py 由 LLM 产出数据；
本脚本用确定性规则模拟同样的数据结构，所有随机数固定种子可复现。

注意：demo.db 中的分析结果为规则生成，并非 LLM 输出，仅用于演示。
"""
import json
import os
import random
import re
import shutil

import numpy as np

from src.config import load_config
from src.rag import EmbeddingConfig, RAG, _to_blob
from src.storage import Storage

# 演示用实体词表：覆盖电商与消费主题常见公司/产品
KNOWN_ENTITIES = [
    "拼多多", "淘宝", "京东", "抖音", "快手", "小红书", "美团",
    "小米", "华为", "荣耀", "苹果", "iPhone", "沃尔玛", "Steam", "吉利",
]
# 演示主题分类：同一主题的文章共享向量基底，演示时自然形成关联脉络
THEME_KEYWORDS = [
    ("财报业绩", ("财报", "营收", "利润", "业绩", "亏损", "盈利")),
    ("新品发售", ("发售", "上市", "推出", "上架", "开售")),
    ("手机市场", ("手机", "iPhone", "华为", "小米", "荣耀", "MagicOS")),
    ("电商零售", ("电商", "零售", "沃尔玛", "超市", "拼多多", "淘宝", "线上销售")),
    ("汽车产业", ("车展", "汽车", "SUV", "吉利", "路虎", "揽胜")),
]
_POSITIVE = ("增长", "超预期", "创新高", "发布", "上市", "发售", "领跑", "升级", "优惠")
_NEGATIVE = ("下降", "暴跌", "亏损", "裁员", "投诉", "处罚", "翻车", "下架", "退")
_DIM = 64  # 演示向量维度


def _sentiment(text: str, rng: random.Random) -> tuple:
    """规则情感判断：负面词优先（坏消息通常更"重要"），置信度带确定性抖动。"""
    if any(w in text for w in _NEGATIVE):
        return "负面", round(0.7 + rng.random() * 0.25, 2)
    if any(w in text for w in _POSITIVE):
        return "正面", round(0.65 + rng.random() * 0.3, 2)
    return "中性", round(0.5 + rng.random() * 0.2, 2)


def _pseudo_vector(entity: str, rng: np.random.RandomState) -> np.ndarray:
    """伪向量：同一实体的文章共享基底向量 + 小噪声，保证演示时同实体文章互相关联。

    注意必须用零均值分布（randn 而非 rand）：[0,1] 均匀分布的向量全落在
    正象限，两两余弦天然高达 0.75 左右，会制造大量假关联。
    """
    if entity:
        base_rng = np.random.RandomState(abs(hash(entity)) % (2**31))
        base = base_rng.randn(_DIM).astype(np.float32)
        vec = 0.9 * base + 0.1 * rng.randn(_DIM).astype(np.float32)
    else:
        vec = rng.randn(_DIM).astype(np.float32)  # 无实体文章随机向量，不会与谁相似
    return (vec / np.linalg.norm(vec)).astype(np.float32)


def _real_embeddings(texts: list, cfg: dict):
    """若配置了 EMBEDDING_API_KEY，用硅基流动 bge 模型批量生成真实 embedding。

    返回 np.float32 向量列表（维度由模型决定，bge-small-zh 为 512）；
    未配置 key 时返回 None，调用方退回伪向量。
    """
    if not os.environ.get("EMBEDDING_API_KEY"):
        return None
    emb_cfg = EmbeddingConfig(cfg)
    if not emb_cfg.available:
        return None
    try:
        rag = RAG(emb_cfg, cfg["industry"]["name"])
        return [np.asarray(v, dtype=np.float32) for v in rag.embed_texts(texts)]
    except Exception as e:  # 接口异常时不致命，退回伪向量
        print(f"[WARN] 真实 embedding 失败（{type(e).__name__}），改用伪向量: {e}")
        return None


def main():
    cfg = load_config()
    src_db = cfg["storage"]["db_path"]
    demo_db = re.sub(r"\.db$", ".demo.db", src_db)
    shutil.copyfile(src_db, demo_db)

    storage = Storage(demo_db)
    rows = storage.conn.execute("SELECT id, title, summary FROM articles ORDER BY id").fetchall()
    keywords = cfg["industry"]["keywords"]
    noise_rng = np.random.RandomState(42)

    # 真实 embedding（需 EMBEDDING_API_KEY）；无 key 时为 None，下方退回伪向量
    real_vecs = _real_embeddings(
        [f"{r['title']} {r['summary'] or ''}" for r in rows], cfg
    )

    # 1. 规则生成分析结果 + 向量（真实或伪）
    first_entity = {}
    for i, r in enumerate(rows):
        text = f"{r['title']} {r['summary'] or ''}"
        # 主主题用于向量关联：同主题文章共享向量基底
        theme = next((name for name, kws in THEME_KEYWORDS if any(k in text for k in kws)), "")
        first_entity[r["id"]] = theme

        # 已有真实 LLM 分析的文章跳过——真实结果优先，规则数据只补齐缺口
        has_real = storage.conn.execute(
            "SELECT 1 FROM analysis WHERE article_id = ?", (r["id"],)
        ).fetchone()
        if not has_real:
            rng = random.Random(r["id"])  # 每篇文章种子固定，结果可复现
            entities = [e for e in KNOWN_ENTITIES if e in text][:3]
            tags = [k for k in keywords if k in text][:3] or [cfg["industry"]["name"]]
            sentiment, conf = _sentiment(text, rng)
            summary_ai = re.split(r"[。！？]", r["summary"] or r["title"])[0][:80]
            result = {
                "summary": summary_ai,
                "sentiment": sentiment,
                "confidence": conf,
                "tags": tags,
                "entities": entities,
            }
            storage.save_analysis(r["id"], result, tokens=0)

        vec = real_vecs[i] if real_vecs is not None else _pseudo_vector(theme, noise_rng)
        storage.save_embedding(r["id"], _to_blob(vec))

    # 2. 相似事件关联（复用 rag 的余弦逻辑，阈值与线上一致）
    threshold = cfg["embedding"]["similarity_threshold"]
    top_k = cfg["embedding"]["top_k"]
    all_emb = storage.embedded_articles()
    mat = {a["id"]: np.frombuffer(a["embedding"], dtype=np.float32) for a in all_emb}
    titles = {a["id"]: a["title"] for a in all_emb}
    for a in all_emb:
        sims = []
        for other in all_emb:
            if other["id"] >= a["id"]:
                continue  # 只关联历史（id 更小）
            sim = float(mat[a["id"]] @ mat[other["id"]])  # 已归一化，点积即余弦
            if sim >= threshold:
                sims.append({"id": other["id"], "title": titles[other["id"]],
                             "similarity": round(sim, 4)})
        sims.sort(key=lambda x: x["similarity"], reverse=True)
        storage.save_related(a["id"], sims[:top_k])

    # 3. 按实体聚类生成趋势脉络（规则模板，模拟 LLM 产出格式）
    from collections import defaultdict
    from datetime import datetime, timedelta

    by_entity = defaultdict(list)
    for r in rows:
        if first_entity[r["id"]]:
            by_entity[first_entity[r["id"]]].append(r["title"])
    week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).date().isoformat()
    n_trends = 0
    for entity, ts in sorted(by_entity.items(), key=lambda kv: -len(kv[1])):
        if len(ts) < 2:
            continue  # 单条事件不构成"脉络"
        ids = [r["id"] for r in rows if r["title"] in ts]
        content = (
            f"本周「{entity}」主题相关动态共 {len(ts)} 条，事件之间呈现明显的连续演进："
            f"从「{ts[0][:20]}…」到「{ts[-1][:20]}…」，反映出该主题在行业内的持续热度，"
            f"值得后续跟踪其下一步进展。"
        )
        storage.save_trend(week_start, [{"id": i, "title": t} for i, t in zip(ids, ts)], content)
        n_trends += 1

    analyzed = storage.conn.execute("SELECT COUNT(*) AS n FROM analysis").fetchone()["n"]
    related = storage.conn.execute(
        "SELECT COUNT(*) AS n FROM articles WHERE related_event IS NOT NULL AND related_event != '[]'"
    ).fetchone()["n"]
    storage.close()
    print(f"演示快照已生成: {demo_db}")
    print(f"分析 {analyzed} 条 | 关联 {related} 条 | 趋势 {n_trends} 条")


if __name__ == "__main__":
    main()
