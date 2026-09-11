# 评测体系说明

> 原则：每个能力都有可复现的验证方式；没有验证的结论明确标注为"待补/演示数据"。

## 1. 单元测试（离线、可复跑）

```bash
cd intel-agent
pytest -q
```

**197 个用例全绿**（告警 25 / 分析 20 / 配置 30 / 采集 20 / 事件 13 / 日志 4 / 邮件 17 / RAG 23 / 日报 15 / 存储 30），覆盖十大模块：

- 配置：六段字段级校验（keywords / sources / max_items_per_source / similarity_threshold / api_key_env 等），错误消息带字段名与段名，入口 1 秒内报错
- 采集：关键词命中过滤（含大小写不敏感匹配、空关键词边界）、标题相似度去重（长度剪枝 + 字符集初筛 + SequenceMatcher 阈值判定）、HTML 清洗（含实体反转义）、HTTP 请求重试、信源健康探测与连续故障熔断跳过机制
- 存储：建表、WAL 模式与防锁库设置、URL 去重、待分析筛选、`save_analysis` 事务回滚、`insert_articles` 批量事务写入、`prune_history` TTL 级联修剪与 WAL Checkpoint、`feed_health` 失败追踪与熔断判定、embedding 存取、related_event / trends 读写、sent_reports 发送记录、sent_alerts 告警记录
- 分析：prompt 构造、JSON 宽容解析（坏 JSON / 缺字段 / 非法枚举 / 置信度钳制）、解析失败重试、指数退避、dry-run 不调 API
- RAG：BLOB 往返、余弦相似度（含零向量防除零）、阈值过滤、Top-K、批量切片向量化与单批重试、幂等关联、周度趋势代表性成员上限（10 篇）防 Token 膨胀、周度趋势无 Key / 异常降级（孤立单篇跳过）、dry-run
- 事件：related_event 链合并（链 / 环 / 去重 / 链尾文章纳入）、长链传递性防漂移与 `max_size=15` 拓扑关联密度截断、按规模排序、时间线排序、窗口过滤、裸 sqlite 连接兼容
- 日志：统一标准 logging 体系、控制台输出与 5MB 文件自动轮转落盘至 `output/agent.log`
- 告警：负面 + 置信度阈值筛选、近 2 天时间窗口限制、sent_alerts 持久化去重防风暴（--force-alert 强制重发）、钉钉 / 企业微信 payload 构造、HTTP/errcode 状态码排障日志留痕、推送成功 / 失败降级、`--webhook-dry-run`
- 日报：按日汇总、情感分布、重要度排序、HTML 渲染（标题/摘要/情感标签/链接转义/空数据/降级说明）、纯文本备用正文渲染
- 邮件：SMTP 配置判定（兼容 `SMTP_TO` 与 `MAIL_TO`）、SSL / STARTTLS 发送、`MIMEMultipart("alternative")` 多部分邮件、`--no-send` 无配置不崩、降级说明写入日报、`_rotate_html_reports` 历史 HTML 报表轮转清理

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
| `python run_daily.py --webhook <url> --webhook-dry-run` | 事件聚合 + 告警 payload | 通过，不实际发送 |
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
- 拿到 key 后按以下命令产出真实数据，跑完回填本节计数：

  ```bash
  export EMBEDDING_API_KEY=sk-xxx   # 必填：embedding 服务（bge-m3，阈值 0.6）
  export LLM_API_KEY=sk-xxx         # 可选：趋势段落走真实 LLM；缺省降级为标题拼接
  python run_rag.py
  ```

- 摘要质量未做人工抽检打分样本量，面试时如实说明评估现状与后续方案

## 6. 评测相关文件

- `tests/test_crawler.py`：单元测试
- `data/intel.db`：真实主库（.gitignore 排除）
- `data/intel.demo.db`：演示快照（提交进仓库）
- `output/daily_*.html`：日报产物
