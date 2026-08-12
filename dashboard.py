"""Streamlit 情报看板：指标总览 / 情感趋势 / 情报筛选 / 趋势脉络 / 对话式问答。

运行：streamlit run dashboard.py
说明：除「立即采集」和「关键词配置」外，看板对数据库只读。
"""
import json
import os
import sqlite3
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.analyzer import Analyzer, LLMConfig
from src.config import load_config
from src.rag import RAG, EmbeddingConfig, _from_blob, cosine
from src.storage import Storage

st.set_page_config(page_title="行业情报看板", page_icon="📊", layout="wide")

# 配色约定：情感分析场景用 绿=正面 / 灰=中性 / 红=负面（语义色，非股票涨跌色）
SENTIMENT_COLOR = {"正面": "#2e9e5b", "中性": "#8a8f98", "负面": "#d64545"}
ACCENT = "#1f6feb"

cfg = load_config()
DB_PATH = cfg["storage"]["db_path"]


@st.cache_data(ttl=60)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """只读查询。新建连接而非复用 Storage，避免 Streamlit 缓存持有可写连接。"""
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


# ---------------- 侧边栏 ----------------

st.sidebar.title("⚙️ 控制台")
st.sidebar.caption(f"行业主题：{cfg['industry']['name']}")

days = st.sidebar.slider("数据时间范围（天）", 1, 30, 7)
since = (date.today() - timedelta(days=days - 1)).isoformat()

st.sidebar.subheader("关注关键词")
kw_text = st.sidebar.text_area(
    "每行一个，保存后立即生效", value="\n".join(cfg["industry"]["keywords"]), height=180
)
if st.sidebar.button("保存关键词"):
    keywords = [k.strip() for k in kw_text.splitlines() if k.strip()]
    if keywords:
        cfg["industry"]["keywords"] = keywords
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        st.sidebar.success(f"已保存 {len(keywords)} 个关键词")
        st.cache_data.clear()
    else:
        st.sidebar.warning("关键词不能为空")

if st.sidebar.button("🔄 立即采集", use_container_width=True):
    from src.crawler import run_crawl

    storage = Storage(DB_PATH)
    try:
        with st.sidebar.status("正在采集...", expanded=True) as status:
            stats = run_crawl(cfg, storage)
            status.update(label="采集完成", state="complete")
        st.sidebar.write(
            f"抓取 {stats['fetched']} 条 | 命中 {stats['matched']} 条 | 新入库 {stats['inserted']} 条"
        )
        if stats["errors"]:
            st.sidebar.warning("；".join(stats["errors"]))
    finally:
        storage.close()
    st.cache_data.clear()

# ---------------- 数据准备 ----------------

articles = query(
    """SELECT a.id, a.title, a.url, a.source, a.published, a.fetched_at,
              a.related_event, an.summary_ai, an.sentiment, an.sentiment_conf,
              an.tags, an.entities
       FROM articles a LEFT JOIN analysis an ON an.article_id = a.id
       WHERE date(a.fetched_at) >= ?
       ORDER BY a.id DESC""",
    (since,),
)
total_all = query("SELECT COUNT(*) AS n FROM articles").iloc[0]["n"]
today_new = query(
    "SELECT COUNT(*) AS n FROM articles WHERE date(fetched_at) = date('now','localtime')"
).iloc[0]["n"]
report_count = query("SELECT COUNT(*) AS n FROM trends").iloc[0]["n"]

# ---------------- 页面主体 ----------------

tab_board, tab_chat = st.tabs(["📊 情报看板", "💬 情报问答"])

with tab_board:
    st.title(f"📊 {cfg['industry']['name']} · 行业情报看板")

    analyzed = articles.dropna(subset=["sentiment"])
    pos = (analyzed["sentiment"] == "正面").sum()
    neg = (analyzed["sentiment"] == "负面").sum()
    ratio = f"{pos}/{neg}" if len(analyzed) else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("今日新增", int(today_new))
    c2.metric("库内总量", int(total_all))
    c3.metric("正面/负面", ratio)
    c4.metric("趋势报告", int(report_count))

    # 情感趋势折线图
    st.subheader("情感趋势")
    if len(analyzed):
        trend = (
            analyzed.assign(day=analyzed["fetched_at"].str[:10])
            .groupby(["day", "sentiment"])
            .size()
            .reset_index(name="n")
        )
        fig = go.Figure()
        for s in ("正面", "中性", "负面"):
            d = trend[trend["sentiment"] == s]
            fig.add_trace(
                go.Scatter(
                    x=d["day"], y=d["n"], name=s, mode="lines+markers",
                    line=dict(color=SENTIMENT_COLOR[s]),
                )
            )
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("该时间范围内暂无已分析数据，先运行 run_analyze.py 生成分析结果。")

    # 情报列表（可筛选）
    st.subheader("情报列表")
    f1, f2, f3 = st.columns(3)
    src_opt = ["全部"] + sorted(articles["source"].dropna().unique().tolist())
    src_sel = f1.selectbox("来源", src_opt)
    sent_sel = f2.selectbox("情感", ["全部", "正面", "中性", "负面", "未分析"])
    kw_sel = f3.text_input("关键词过滤")

    view = articles.copy()
    if src_sel != "全部":
        view = view[view["source"] == src_sel]
    if sent_sel == "未分析":
        view = view[view["sentiment"].isna()]
    elif sent_sel != "全部":
        view = view[view["sentiment"] == sent_sel]
    if kw_sel.strip():
        view = view[view["title"].str.contains(kw_sel.strip(), case=False, na=False)]
    st.caption(f"共 {len(view)} 条")

    for _, r in view.head(50).iterrows():
        sentiment = r["sentiment"] if pd.notna(r["sentiment"]) else "未分析"
        color = SENTIMENT_COLOR.get(sentiment, "#c0c4cc")
        with st.container(border=True):
            st.markdown(
                f"**[{r['title']}]({r['url']})** "
                f"<span style='background:{color};color:white;padding:1px 8px;"
                f"border-radius:10px;font-size:12px'>{sentiment}</span>",
                unsafe_allow_html=True,
            )
            if pd.notna(r["summary_ai"]):
                st.write(r["summary_ai"])
            meta = f"{r['source']} · {r['fetched_at'][:16]}"
            if pd.notna(r["tags"]):
                meta += " · " + " / ".join(json.loads(r["tags"]))
            st.caption(meta)
            if pd.notna(r["related_event"]) and r["related_event"] != "[]":
                rel = json.loads(r["related_event"])
                if rel:
                    st.caption("🔗 关联历史：" + "；".join(f"《{x['title']}》" for x in rel))

    # 趋势脉络
    st.subheader("趋势脉络")
    trends = query("SELECT * FROM trends ORDER BY id DESC LIMIT 5")
    if len(trends):
        for _, t in trends.iterrows():
            with st.container(border=True):
                st.markdown(f"**{t['week_start']} 当周趋势**")
                st.write(t["content"])
    else:
        st.info("暂无趋势数据，运行 run_rag.py 后生成。")

# ---------------- 对话式问答 ----------------

with tab_chat:
    st.title("💬 情报问答")
    st.caption("基于库内情报的检索增强问答（RAG），回答附引用来源。")

    emb_cfg = EmbeddingConfig(cfg)
    llm_cfg = LLMConfig(cfg)
    llm_ready = llm_cfg.available
    if not llm_ready:
        st.warning("未配置 LLM，当前为纯检索模式：返回相关情报列表，不生成自然语言回答。")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    def retrieve(question: str, top_k: int = 5) -> pd.DataFrame:
        """优先向量检索；embedding 接口不可用时自动降级为关键词匹配。"""
        if emb_cfg.available:
            try:
                rag = RAG(emb_cfg, cfg["industry"]["name"])
                qvec = rag.embed_texts([question])[0]
                rows = query(
                    "SELECT id, title, url, summary, embedding FROM articles WHERE embedding IS NOT NULL"
                )
                if len(rows):
                    matrix = np.stack([_from_blob(b) for b in rows["embedding"]])
                    sims = cosine(qvec, matrix)
                    rows = rows.assign(sim=sims).sort_values("sim", ascending=False)
                    return rows.head(top_k)
                return rows
            except Exception as e:
                # 典型情况：LLM_API_KEY 被误当作 embedding key，硅基流动接口 401
                st.warning(f"向量检索不可用（{type(e).__name__}），已降级为关键词匹配。")
        # 关键词兜底：任一查询词命中标题/摘要即返回
        mask = pd.Series(False, index=articles.index)
        for w in question.split():
            mask |= articles["title"].str.contains(w, case=False, na=False)
            if "summary" in articles.columns:
                mask |= articles["summary"].fillna("").str.contains(w, case=False, na=False)
        return articles[mask].head(top_k)

    if question := st.chat_input("提问，例如：这周拼多多有什么动态？"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        hits = retrieve(question)
        refs = "\n".join(
            f"- [{r['title']}]({r['url']})" for _, r in hits.iterrows() if pd.notna(r.get("url"))
        )

        if llm_ready and len(hits):
            context = "\n".join(
                f"《{r['title']}》：{(r.get('summary') or '')[:200]}" for _, r in hits.iterrows()
            )
            history_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-6:-1]
            )
            prompt = (
                f"你是行业情报助手。基于以下检索到的情报回答用户问题，"
                f"回答结尾不要自行编造来源。\n\n相关情报：\n{context}\n\n"
                f"对话历史：\n{history_text}\n\n用户问题：{question}"
            )
            from openai import OpenAI

            client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)
            answer = client.chat.completions.create(
                model=llm_cfg.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, timeout=60,
            ).choices[0].message.content
            reply = answer + ("\n\n**引用来源：**\n" + refs if refs else "")
        elif len(hits):
            reply = "**检索到以下相关情报（未配置 LLM）：**\n" + (refs or "（无）")
        else:
            reply = "库内未检索到相关情报，可先点击侧边栏「立即采集」补充数据。"

        with st.chat_message("assistant"):
            st.markdown(reply, unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
