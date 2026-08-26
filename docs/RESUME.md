# 简历条目（写实版）

> 原则：只写真实存在的能力；需要真实数据的地方标注 [待补]，拿到 Key 后填上。
> 数字比形容词有说服力：3 个 RSS 源、10 个测试、34 条情报、双重去重 0% 重复入库……

## 中文版

**智能行业情报分析 Agent — 自动化情报流水线** ｜ 独立开发 ｜ 2026.08

- 独立实现行业情报自动化流水线：RSS 采集（36氪/IT之家/少数派）→ 关键词过滤 → **双重去重**（URL 唯一约束 + 标题相似度 ≥0.85 事件级去重）→ SQLite 存储；
- 基于 DeepSeek 实现 LLM 结构化分析（摘要/情感/标签/实体）：prompt 字段契约 + JSON 强制输出 + 宽容提取严格校验 + 失败重试，实测单次上限 20 篇、token 用量入库可核算成本；
- 实现 **RAG 历史情报关联**：bge-m3 embedding 以 BLOB 存 SQLite、numpy 全量余弦相似度检索 Top-3 相关事件 + 周度趋势聚类，说明语义相似 ≠ 字面相似；数据量小时不引入向量数据库，避免过度工程；
- 交付 Streamlit 可视化看板（指标卡/情感趋势/情报列表/**对话式问答带引用**）+ HTML 邮件日报 + Windows 定时任务，全部模块支持 `--dry-run` 与自动降级（无 Key 不崩）；
- 工程化：**10 个 pytest 用例全绿**、增量采集实测重复入库率 0%、日报全流程 < 30 秒、演示快照支持云端零 Key 部署（https://intel-agent.streamlit.app）。

## English Version

**Intelligent Industry Intelligence Agent — Automated News Pipeline** | Solo Developer | 2026.08

- Built an automated industry-intelligence pipeline: RSS ingestion (36Kr / ITHome / Sspai) → keyword filtering → dual-layer dedup (URL uniqueness + ≥0.85 title similarity as event-level granularity) → SQLite persistence;
- Implemented structured LLM analysis (summary / sentiment / tags / entities) on DeepSeek with a strict JSON contract, lenient-parse/strict-validate, and retry-on-failure; per-run cap of 20 articles with token usage tracked for cost control;
- Implemented RAG-based historical-event association: bge-m3 embeddings stored as BLOBs in SQLite, numpy full cosine similarity for top-3 related events plus weekly trend clustering; deliberately avoided a vector database at this scale (no over-engineering);
- Delivered a Streamlit dashboard (metric cards / sentiment trends / feed / chat Q&A with citations) and HTML email reports with Windows scheduled tasks; every module supports --dry-run and graceful degradation without API keys;
- Engineering: 10 passing pytest cases, measured 0% duplicate ingestion, full daily pipeline under 30s, and a demo snapshot for zero-key cloud deployment (https://intel-agent.streamlit.app).

## 面试一句话总结（30 秒电梯演讲）

「我做了一个行业情报分析 Agent：每天自动从三个 RSS 源抓电商与消费行业的资讯，
经过双重去重、LLM 结构化分析和 RAG 历史关联，生成可视化看板加每日邮件日报。
我重点讲了两个选型决策——数据源为什么选 RSS、小数据量为什么不上向量数据库——
而且整个项目每个模块都能 dry-run、能降级、能算成本，工程闭环是完整的。」

## 待补数据

- [ ] 真实 RAG 关联 / 趋势结果（配置 `EMBEDDING_API_KEY` 跑 `run_rag.py` 后更新）
- [ ] 在线 Demo 可访问后的截图（dashboard / chat / email）
- [ ] GitHub 仓库链接（推送后替换）

