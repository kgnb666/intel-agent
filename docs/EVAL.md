# 评测体系说明

> 原则：每个能力都有可复现的验证方式；没有验证的结论明确标注为"待补/演示数据"。

## 1. 单元测试（离线、可复跑）

```bash
cd intel-agent
pytest -q
```

**10 个用例全绿**，覆盖：

- 关键词命中过滤（含大小写 / 空关键词边界）
- 标题相似度去重（SequenceMatcher 阈值判定）
- HTML 清洗
- HTTP 请求重试机制（重试耗尽抛错、成功即返回）

## 2. 采集与去重（实测记录）

- 3 个 RSS 源单轮抓取约 60 条，关键词命中约 12 条
- 双重去重后入库：实测重复入库率约 0%（同一链接绝不重复）
- 增量验证：首轮入库后二轮再跑 0 新增
- 当前主库 `data/intel.db`：**34 条文章 / 24 条已分析**

复现：`python run_crawl.py` 连续跑两轮，观察第二轮新增数。

## 3. 流水线可用性（dry-run 全链路）

| 命令 | 验证内容 | 结论 |
|---|---|---|
| `python run_analyze.py --dry-run` | prompt 构造与流程 | 通过 |
| `python run_rag.py --dry-run` | 向量化/检索流程 | 通过 |
| `python run_daily.py --no-send` | 日报生成 + 降级容错 | 通过，产出 HTML |
| `streamlit run dashboard.py` | 看板 HTTP 200 | 通过 |

无 API Key 也能验证全部流程——这是"工程可用"的一部分。

## 4. 性能与成本（实测口径）

- 日报生成到 HTML 落盘全流程 < 30 秒（无 LLM 调用时）
- 单次 LLM 分析上限 `max_per_run=20` 篇（config 可配）
- 每篇分析 token 用量入库（`tokens_used`），可追溯每日成本
- embedding 使用硅基流动免费模型，向量环节零成本
- 百级数据量下 numpy 全量余弦检索为毫秒级

## 5. 诚实边界（重要）

- **RAG 关联 / 趋势数据目前只有演示快照**：`intel.demo.db` 中的关联与趋势由 `make_demo_snapshot.py` 用确定性规则生成（24 篇全分析 / 3 条趋势），**并非真实 LLM 输出**；主库 `intel.db` 的 `related_event` / `trends` 尚未用真实 embedding 跑出结果
- 配置 `EMBEDDING_API_KEY` 后运行 `python run_rag.py` 即可产出真实关联（bge-m3，阈值 0.6）
- 摘要质量未做人工抽检打分样本量，面试时如实说明评估现状与后续方案

## 6. 评测相关文件

- `tests/test_crawler.py`：单元测试
- `data/intel.db`：真实主库（.gitignore 排除）
- `data/intel.demo.db`：演示快照（提交进仓库）
- `output/daily_*.html`：日报产物

