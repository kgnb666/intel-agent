# 投递前检查清单

> 目标：投递前把每一栏都打勾。已打勾的是代码里确认过的，剩下的是需要在自己机器/服务器上操作的。

## ✅ 代码与质量（已完成，可复跑验证）

- [x] 单元测试全绿：`cd intel-agent && pytest -q`（197 个，覆盖十大模块）
- [x] 静态代码检查：`ruff check src tests *.py` 0 警告 0 错误通过 CI 门禁
- [x] 容器化交付物：`Dockerfile`、`.dockerignore` 与 `docker-compose.yml` 齐备
- [x] 采集实测：3 源单轮约 60 条、命中约 12 条、重复入库 0%
- [x] 主库数据：`data/intel.db` 34 条文章 / 24 条已分析
- [x] dry-run 全链路：`run_analyze --dry-run` / `run_rag --dry-run` / `run_daily --no-send` 均通过
- [x] 事件时间线 + 告警 dry-run：`run_daily --webhook <url> --webhook-dry-run` 通过（不实际发送）
- [x] 看板启动正常：`streamlit run dashboard.py` → HTTP 200
- [x] 演示快照：`make_demo_snapshot.py` 产出 `intel.demo.db`（24 篇分析 / 3 条趋势）

## 🚀 部署上线

- [ ] 推送 GitHub：`git init && git add . && git commit -m "feat: intel-agent 全链路" && git push`
- [ ] 确认 `.gitignore` 排除 `data/intel.db`（真实库）与 `.env`；只提交 `intel.demo.db`
- [ ] Streamlit Cloud：选择仓库 + 入口 `dashboard.py`，配好 Secrets（`INTEL_DB_PATH` / `LLM_API_KEY` / `EMBEDDING_API_KEY`）
- [ ] 验证 https://intel-agent.streamlit.app 可访问、问答降级路径正常
- [ ] 有 Key 时跑 `python run_rag.py` 补真实关联数据，把结果更新到 docs/EVAL.md / docs/RESUME.md

## 🎬 演示与求职包装

- [ ] 按 `docs/DEMO_SCRIPT.md` 录 3 分钟演示视频
- [ ] 补三张截图（dashboard / chat / email）到 `docs/screenshots/`
- [ ] 视频上传 B 站/YouTube，链接放简历与 README
- [ ] 按 `docs/RESUME.md` 套进简历（含 Demo 链接 + GitHub 链接）
- [ ] 面试问答预案 10 问全部能口头讲一遍（`面试问答预案.md`）
- [ ] 30 秒电梯演讲背熟

## 🧪 投递前最后自测（模拟面试官）

- [ ] 干净机器走一遍：采集 → 分析 → 关联 → 看板 → 日报
- [ ] 问自己：「为什么用 RSS」「双重去重怎么做的」「为什么用向量检索」「数据量大了怎么办」「最难的地方是什么」
- [ ] 确认 API Key 没有出现在任何提交文件里
