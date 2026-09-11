# 开发路线图（回顾版）

> 目标：做一个能串起**爬虫、数据库、LLM 应用、RAG、前端可视化、定时任务**的完整工程闭环，每个阶段都有可见成果。以下按实际完成顺序回顾。

## 阶段 0：选题与骨架（第 1 天）

- [x] 确定方向：AI 应用「智能行业情报分析 Agent」，行业定为电商/消费（entry-level 岗位池最大）
- [x] 搭好项目骨架：config.yaml（行业/关键词/数据源可配置）+ `src/crawler.py` + `src/storage.py` + `run_crawl.py`
- [x] 首轮实测：抓取 50 条、命中 10 条入库；二轮 0 新增（增量去重验证通过）
- [x] 数据源定稿：36氪 / IT之家 / 少数派（虎嗅 RSS 持续超时已移除）

**里程碑**：能跑通"采集 → 关键词过滤 → 去重入库"。

## 阶段 1：LLM 结构化分析（第 6~8 天）

- [x] `src/analyzer.py`：JSON 字段契约 + `response_format=json_object` + 宽容提取严格校验 + 重试 2 次
- [x] `analysis` 表：摘要 / 情感 / 标签 / 实体 / token 用量
- [x] 成本控制：`max_per_run=20`、`analyzed` 增量标记
- [x] 容错：API 超时/限流指数退避，单篇失败不阻断整批
- [x] `run_analyze.py` 支持 `--dry-run`

**里程碑**：无 Key 也能验证分析流程；有 Key 产出真实结构化分析。

## 阶段 2：RAG 历史情报关联（第 8~9 天）

- [x] `src/rag.py`：embedding 向量化（BLOB 存 SQLite）+ numpy 全量余弦 + Top-3 相似事件
- [x] 阈值调优：bge-m3 语义相似度偏低，关联阈值定为 0.6（质量/数量平衡点）
- [x] `related_event` 关联链 + `trends` 周度趋势聚类
- [x] `run_rag.py` 支持 `--dry-run`；无 Key 时看板问答自动降级关键词检索

**里程碑**：能讲清"为什么用 RAG、为什么不上向量数据库"。

## 阶段 3：Streamlit 可视化看板（第 9~11 天）

- [x] 顶部指标卡 + 情感趋势折线图（plotly）+ 情报列表（筛选/情感标签）
- [x] 对话式问答 Tab（复用 RAG 检索，带引用来源）
- [x] 一键采集按钮 + 关键词在线配置
- [x] 云端无库自动回退演示库、漏装 pandas 补依赖

**里程碑**：看板即"脸面"，无 Key 也能完整演示。

## 阶段 4：日报与邮件推送（第 11~12 天）

- [x] `src/report.py`：HTML 日报（table 布局兼容邮件客户端）
- [x] `src/mailer.py`：SMTP 推送，`--no-send` 只落盘
- [x] `run_daily.py`：一键流水线 + 每步独立容错 + 降级说明写进邮件
- [x] Windows 任务计划程序定时执行方案

**里程碑**：`run_daily.py --no-send` 产出日报 HTML，全流程 < 30 秒。

## 阶段 5：打磨与交付（8-24 前后）

- [x] `make_demo_snapshot.py` 生成演示快照（24 篇全分析 / 3 条趋势）
- [x] 看板云端自动回退演示库（`no such table` 崩溃修复）
- [x] requirements 补显式 pandas；真实 bge-m3 embedding 支持
- [x] 197 个单元测试全绿；README / PROMPTS / 面试问答预案齐备
- [x] 部署实例：https://intel-agent.streamlit.app

**里程碑**：项目可交付、可演示、可讲。

## 阶段 6：事件聚合与实时告警（9-01 前后）

- [x] `src/events.py`：related_event 关联链 → 事件实体（事件 id / 标题 / 时间线 / 文章列表），并查集求连通分量，链、环、重复边自动去重
- [x] dashboard 新增「事件时间线」视图：同一事件的多篇报道按时间展开
- [x] `src/alert.py`：负面且置信度 ≥ 0.8 的事件推送钉钉/企业微信 webhook（`--webhook` / `alert.webhook` / `ALERT_WEBHOOK`）
- [x] 告警降级：无 webhook 配置静默跳过；推送失败只记降级说明、不中断流水线；`--webhook-dry-run` 可无副作用预览

## 阶段 9：业务稳定性与告警防灾加固（9-11）

- [x] `sent_alerts` 持久化表与去重机制：以 `(event_id, article_id)` 唯一复合键防范流水线重试与高频调度的告警风暴
- [x] Webhook 推送失败诊断留痕：HTTP 状态码、errcode 与响应体正文 warning 级别精准排障日志
- [x] 看板采集与问答并发保护：全局线程锁 `CRAWL_LOCK` 防止重复点击抓取冲突，智能问答 3 秒提交冷却防抖

## 阶段 10：性能伸缩、数据生命周期与健康监控（9-11）

- [x] `prune_history` TTL 级联修剪机制：按 `--prune-days` 自动清理超期文章、分析详情及已推送告警记录，并执行 `PRAGMA wal_checkpoint(TRUNCATE)` 阻断存储膨胀
- [x] HTML 历史报表自动轮转清理：`_rotate_html_reports` 默认保留最新 30 份报表，防止磁盘碎片堆积
- [x] 采集层 `insert_articles` 批量事务写入：单次抓取批量单事务 Commit，彻底解决单条频密提交造成的 I/O 损耗与锁冲突
- [x] RSS 信源健康度追踪与故障熔断机制：`feed_health` 统计连续失败次数与错误摘要，连续失败 ≥5 次触发熔断跳过，保障流水线稳定高效

## 阶段 11：交互体验深度优化与工程化标准交付（9-11）

- [x] 看板全量向量检索规模保护：限定 90 天/3000 条匹配范围，避免数据量扩大导致瞬时反序列化爆内存
- [x] 看板事件时间线查询缓存：抽离 `load_cached_events` 配合 `@st.cache_data(ttl=60)`，滑块拖动响应提速至毫秒级
- [x] 事件并查集长链传递性防漂移：`aggregate_events` 增加 `max_size=15` 与相似度关联密度拓扑剪枝，阻断弱相关报道膨胀
- [x] 趋势提炼 Prompt 瘦身：`weekly_trends` 限制单趋势簇代表性文章上限 10 篇，节约 Token 并保持核心聚焦
- [x] 生产 Docker 容器化交付物：轻量化 `Dockerfile`、`.dockerignore` 与双服务编排 `docker-compose.yml`
- [x] CI 静态代码质量门禁：集成 `ruff` 扫描（`--select=E,F,W --ignore=E501`），代码库保持 0 规范告警
- [x] 197 个单元测试全绿（覆盖告警/分析/配置/采集/事件/日志/邮件/RAG/日报/存储十大模块）

**里程碑**：内存规模受控、图计算秒级缓存响应、交付容器化、CI 规范门禁全绿。

## 剩余事项

- [ ] 用 `EMBEDDING_API_KEY` 跑真实 RAG，把主库的关联/趋势数据补上（当前为演示快照数据）
- [ ] 扩充信源（公众号 / 微博 / 雪球，需对付反爬）
- [ ] 补充演示截图到 `docs/screenshots/`
