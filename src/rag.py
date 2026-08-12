"""RAG 模块：历史情报向量化关联，形成"趋势脉络"。

设计要点（面试常问）：
- 为什么用向量检索而非关键词匹配：关键词只能命中字面相同的词，
  "拼多多财报超预期"和"多多买菜盈利改善"字面无交集但语义相关，向量能抓住这种关联
- 为什么数据量小不上向量数据库：百级数据量下 numpy 全量算余弦相似度是毫秒级，
  引入 faiss/Milvus 属于过度工程，BLOB 存 float 数组够用且零依赖
"""
import json
import os
from datetime import datetime, timedelta

import numpy as np

_TREND_PROMPT = """你是一名行业情报分析师。以下是「{industry}」行业本周发生的一组相互关联的事件，请用一段话（150 字内）提炼它们共同反映的趋势脉络，指出事件之间的演进关系。

事件列表：
{events}

只输出趋势段落本身，不要输出标题或任何其他文字。"""


class EmbeddingConfig:
    """Embedding 服务配置。DeepSeek 无 embedding 接口，默认走硅基流动免费模型。"""

    def __init__(self, cfg: dict):
        emb = cfg.get("embedding", {})
        self.base_url = os.environ.get("EMBEDDING_BASE_URL") or emb.get(
            "base_url", "https://api.siliconflow.cn/v1"
        )
        self.model = os.environ.get("EMBEDDING_MODEL") or emb.get("model", "BAAI/bge-small-zh-v1.5")
        # 多数平台 embedding 与对话共用 key，缺省时回退读 LLM_API_KEY
        key_env = emb.get("api_key_env", "EMBEDDING_API_KEY")
        self.api_key = os.environ.get(key_env) or os.environ.get("LLM_API_KEY", "")
        self.threshold = float(emb.get("similarity_threshold", 0.75))
        self.top_k = int(emb.get("top_k", 3))

    @property
    def available(self) -> bool:
        return bool(self.api_key)


def _to_blob(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query 与矩阵每一行的余弦相似度。"""
    q_norm = np.linalg.norm(query)
    m_norm = np.linalg.norm(matrix, axis=1)
    denom = q_norm * m_norm
    denom[denom == 0] = 1e-9  # 防零向量除零
    return matrix @ query / denom


class RAG:
    """向量化 + 相似事件检索 + 周度趋势聚类。"""

    def __init__(self, emb_cfg: EmbeddingConfig, industry: str, llm_cfg=None, dry_run: bool = False):
        self.cfg = emb_cfg
        self.industry = industry
        self.llm_cfg = llm_cfg
        self.emb_client = None
        self.chat_client = None
        if not dry_run:
            from openai import OpenAI

            self.emb_client = OpenAI(api_key=emb_cfg.api_key, base_url=emb_cfg.base_url)
            # 趋势提炼复用 analyzer 的对话配置，两者可以是不同供应商
            if llm_cfg and llm_cfg.available:
                self.chat_client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)

    # ---------- 向量化 ----------

    def embed_texts(self, texts: list) -> list:
        resp = self.emb_client.embeddings.create(model=self.cfg.model, input=texts)
        return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

    def vectorize_pending(self, storage, dry_run: bool = False) -> int:
        """给没有向量的文章补 embedding，返回处理数。"""
        pending = storage.articles_without_embedding()
        if not pending:
            return 0
        if dry_run:
            for a in pending:
                print(f"[DRY-RUN] 将向量化: #{a['id']} 《{a['title']}》")
            return len(pending)
        # 批量调用减少网络往返
        texts = [(a["title"] + " " + (a["text"] or ""))[:800] for a in pending]
        vectors = self.embed_texts(texts)
        for a, v in zip(pending, vectors):
            storage.save_embedding(a["id"], _to_blob(v))
        return len(pending)

    # ---------- 相似事件检索 ----------

    def find_related(self, storage, article_id: int, query_vec: np.ndarray) -> list:
        """在历史文章中检索 Top-K 相似事件，低于阈值的不算相关。"""
        history = storage.embedded_articles(before_id=article_id)
        if not history:
            return []
        matrix = np.stack([_from_blob(h["embedding"]) for h in history])
        sims = cosine(query_vec, matrix)
        order = np.argsort(sims)[::-1][: self.cfg.top_k]
        return [
            {
                "id": history[i]["id"],
                "title": history[i]["title"],
                "similarity": round(float(sims[i]), 4),
            }
            for i in order
            if sims[i] >= self.cfg.threshold
        ]

    def relate_pending(self, storage, dry_run: bool = False) -> int:
        """为尚无关联记录的文章检索历史相似事件并入库。"""
        rows = storage.embedded_articles()
        count = 0
        for a in rows:
            cur = storage.conn.execute(
                "SELECT related_event FROM articles WHERE id = ?", (a["id"],)
            ).fetchone()
            if cur["related_event"] is not None:
                continue  # 已关联过，跳过（幂等，可重复运行）
            if dry_run:
                print(f"[DRY-RUN] 将检索关联: #{a['id']} 《{a['title']}》")
                count += 1
                continue
            related = self.find_related(storage, a["id"], _from_blob(a["embedding"]))
            storage.save_related(a["id"], related)
            count += 1
        return count

    # ---------- 周度趋势聚类 ----------

    def weekly_trends(self, storage, dry_run: bool = False) -> list:
        """把本周通过相似度连成链的事件聚类，逐簇让 LLM 提炼趋势段落。

        聚类用并查集：related_event 构成无向边，连通分量即一个趋势簇。
        数据量小，不需要引入图库或聚类算法库。
        """
        week_start = (datetime.now() - timedelta(days=datetime.now().weekday())).date().isoformat()
        rows = storage.conn.execute(
            """SELECT a.id, a.title, a.related_event FROM articles a
               WHERE date(a.fetched_at) >= ? AND a.related_event IS NOT NULL""",
            (week_start,),
        ).fetchall()

        parent = {r["id"]: r["id"] for r in rows}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for r in rows:
            for rel in json.loads(r["related_event"]):
                if rel["id"] in parent:
                    parent[find(r["id"])] = find(rel["id"])

        clusters = {}
        for r in rows:
            clusters.setdefault(find(r["id"]), []).append(r)

        results = []
        for members in clusters.values():
            events = "\n".join(f"- {m['title']}" for m in members)
            prompt = _TREND_PROMPT.format(industry=self.industry, events=events)
            if dry_run:
                print(f"[DRY-RUN] 趋势簇（{len(members)} 条）将发送 prompt:\n{prompt}\n{'-' * 60}")
                continue
            if not self.chat_client:
                # 无对话 key 时降级：直接拼接事件标题作为趋势描述
                content = "；".join(m["title"] for m in members)
            else:
                resp = self.chat_client.chat.completions.create(
                    model=self.llm_cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    timeout=60,
                )
                content = resp.choices[0].message.content.strip()
            cluster_info = [{"id": m["id"], "title": m["title"]} for m in members]
            storage.save_trend(week_start, cluster_info, content)
            results.append({"cluster": cluster_info, "content": content})
        return results

    def run(self, storage, dry_run: bool = False) -> dict:
        """完整 RAG 流程：向量化 -> 关联 -> 趋势。"""
        n_vec = self.vectorize_pending(storage, dry_run=dry_run)
        n_rel = self.relate_pending(storage, dry_run=dry_run)
        trends = self.weekly_trends(storage, dry_run=dry_run)
        return {"vectorized": n_vec, "related": n_rel, "trends": len(trends)}
