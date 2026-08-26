# 项目完成报告

> 本文档总结项目全貌，映射原始目标：**做一个能串起全链路工程能力、可演示、可讲的实习面试项目**。

## 一、项目一句话

**智能行业情报分析 Agent**：每天自动从 36氪 / IT之家 / 少数派抓取电商与消费行业资讯，经双重去重、LLM 结构化分析、RAG 历史关联后，产出可视化看板与每日邮件日报；改一个配置文件即可切换任意行业。

## 二、目标达成情况

| 原始要求 | 达成情况 |
|---|---|
| 爬虫 / 数据采集 | ✅ RSS 采集 + 关键词过滤 + 双重去重（URL 唯一 + 标题相似度 ≥0.85） |
| 数据库 | ✅ SQLite 三表（articles / analysis / trends），存储访问收敛单文件，预留 PostgreSQL 迁移路径 |
| LLM 应用 | ✅ DeepSeek 结构化分析（JSON 契约 + 重试 + 增量 + token 成本核算） |
| RAG | ✅ bge-m3 向量化 + numpy 余弦 Top-3 关联 + 周度趋势聚类，无 Key 自动降级 |
| 前端可视化 | ✅ Streamlit 看板：指标卡 / 情感趋势 / 情报列表 / 对话式问答（带引用） |
| 定时任务 / 推送 | ✅ run_daily 一键流水线 + HTML 日报 + SMTP 邮件 + Windows 任务计划方案 |
| 可演示 | ✅ 演示快照零 Key 部署 + 3 分钟分镜脚本 + 已上线 https://intel-agent.streamlit.app |
| 可量化 | ✅ 10 个单元测试、实测去重率 0%、日报全流程 < 30 秒（见 docs/EVAL.md） |
| 面试准备 | ✅ README 技术决策 + 面试问答预案 10 问 + 简历条目 + 30 秒电梯演讲 |

## 三、工程亮点（面试核心故事线）

1. **数据源选型**：RSS 而非硬爬——稳定、无反爬、结构规范，把精力留给 LLM 应用
2. **双重去重**：URL 约束精确层 + 标题相似度事件层，去重粒度从"链接"提升到"事件"
3. **LLM 输出工程化**：字段契约 + JSON 强制 + 宽容解析严格校验 + 失败留痕，不把 LLM 当确定性函数
4. **选型边界清晰**：小数据量不上向量数据库（numpy 毫秒级、BLOB 零依赖），但检索逻辑独立预留 faiss/Milvus 升级路径
5. **降级文化**：所有模块支持 dry-run / 自动降级 / 每步容错，"半成品日报好过没有日报"

## 四、验证数据（可复现）

```
单元测试      10 个全绿（关键词过滤/相似度去重/HTML 清洗/重试机制）
采集去重      3 源单轮约 60 条 → 命中约 12 条 → 重复入库 0%
主库现状      data/intel.db：34 条文章 / 24 条已分析
演示快照      intel.demo.db：24 篇全分析 / 3 条趋势（规则数据，非 LLM 输出）
全流程耗时    日报生成 < 30 秒（无 LLM 调用时）
```

复现：`pytest -q`、`python run_crawl.py`、`python run_daily.py --no-send`、`streamlit run dashboard.py`。

## 五、技术栈与规模

- 采集：feedparser + requests（超时/重试）；存储：SQLite；分析：DeepSeek（OpenAI 兼容）
- RAG：bge-m3（硅基流动）embedding + numpy 余弦；前端：Streamlit + plotly
- 工程：10 个 pytest 用例、dry-run 全链路、演示快照、Windows 定时任务

## 六、诚实边界与剩余事项

1. **RAG 关联 / 趋势当前只有演示快照数据**（规则生成）；配置 `EMBEDDING_API_KEY` 跑 `run_rag.py` 可补真实结果
2. 摘要质量尚未做人工抽检打分（面试时如实说明评估现状与后续方案）
3. 部署实例已上线，但演示截图未补（见 docs/DEMO_SCRIPT.md）
4. 后续演进：信源扩展 → 事件实体聚合 → 负面事件实时告警

## 七、给用户的投递前提醒

- API Key 只允许通过环境变量注入，`config.yaml` 中只有变量名——推送前 `git grep` 确认无真实 Key
- `data/intel.db` 是真实库，`.gitignore` 已排除；云端部署只提交 `intel.demo.db`

