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
- [x] 10 个单元测试全绿；README / PROMPTS / 面试问答预案齐备
- [x] 部署实例：https://intel-agent.streamlit.app

**里程碑**：项目可交付、可演示、可讲。

## 剩余事项

- [ ] 用 `EMBEDDING_API_KEY` 跑真实 RAG，把主库的关联/趋势数据补上（当前为演示快照数据）
- [ ] 扩充信源（公众号 / 微博 / 雪球，需对付反爬）
- [ ] 事件级聚合：相似标题 → 真正的事件实体与时间线
- [ ] 负面高置信度事件实时告警（微信 / 钉钉 webhook）
- [ ] 补充演示截图到 `docs/screenshots/`

