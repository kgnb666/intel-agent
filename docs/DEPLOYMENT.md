# 部署指南

项目支持三种运行形态：**本机一键流水线**（含 Windows 定时任务）、**Streamlit Community Cloud 在线看板**、以及**自托管服务**。所有敏感值（API Key / SMTP 密码）一律走环境变量，不落配置文件。

## 1. 前置条件

```bash
cd intel-agent
pip install -r requirements.txt
```

## 2. 本机运行（开发 / 演示）

```bash
# 采集（开箱即用，无需任何 key）
python run_crawl.py

# LLM 分析（无 key 用 --dry-run 验证流程）
export LLM_API_KEY=sk-xxx
python run_analyze.py --dry-run
python run_analyze.py

# RAG 关联（DeepSeek 无 embedding，默认走硅基流动；缺省回退读 LLM_API_KEY）
export EMBEDDING_API_KEY=sk-yyy
python run_rag.py

# 日报（不发邮件，HTML 落盘 output/）
python run_daily.py --no-send

# 看板
streamlit run dashboard.py
```

看板无 Key 也能展示：未配置 embedding key 时问答自动降级为关键词检索；数据文件缺失时自动回退演示库 `data/intel.demo.db`。

## 3. 每日定时推送（Windows 任务计划程序）

```bat
schtasks /create /tn "IntelAgentDaily" /tr "\"C:\path\to\venv\Scripts\python.exe\" C:\path\to\intel-agent\run_daily.py" /sc daily /st 08:30 /f
```

创建后在"任务计划程序"GUI 中可查看/修改。API Key 与 SMTP 配置需写入系统环境变量（或任务启动的 bat 脚本中）：

```bat
set LLM_API_KEY=sk-xxx
set EMBEDDING_API_KEY=sk-yyy
set SMTP_HOST=smtp.example.com
set SMTP_PORT=465
set SMTP_USER=you@example.com
set SMTP_PASSWORD=***
set SMTP_TO=receiver@example.com
"C:\path\to\venv\Scripts\python.exe" C:\path\to\intel-agent\run_daily.py
```

## 4. 部署到 Streamlit Community Cloud（推荐，免费）

1. 把项目 push 到 GitHub 公开仓库（`intel-agent/` 作为仓库根目录）。
2. 生成演示数据快照：
   ```bash
   python make_demo_snapshot.py   # 从 data/intel.db 生成 data/intel.demo.db
   ```
   把 `data/intel.demo.db` 提交进仓库。注意：快照里的分析/关联/趋势是**规则生成的演示数据**，并非 LLM 输出；有 Key 的环境应跑 `run_analyze.py` / `run_rag.py` 产出真实数据。
3. 到 share.streamlit.io 用 GitHub 账号登录，选择仓库、入口文件 `dashboard.py`。
4. 在 Settings → Secrets 中配置：
   ```toml
   INTEL_DB_PATH = "data/intel.demo.db"   # 指向演示快照，只读展示
   LLM_API_KEY = "sk-xxx"                 # 可选，不配则问答降级为纯检索
   EMBEDDING_API_KEY = "sk-yyy"           # 可选，不配时向量检索降级关键词匹配
   ```
5. 当前部署实例：https://intel-agent.streamlit.app

## 5. Docker 容器化部署（生产交付）

项目已内置标准生产级 `Dockerfile` 与 `docker-compose.yml`，支持一键拉起看板与自动化抓取流水线：

```bash
# 构建并后台启动看板与定时流水线服务
docker-compose up -d --build

# 查看运行日志
docker-compose logs -f

# 停止服务
docker-compose down
```

挂载卷说明：
- `./data:/app/data`：持久化 SQLite 数据库文件（`intel.db`）。
- `./output:/app/output`：持久化历史邮件日报（`daily_*.html`）与运行日志（`agent.log`）。

## 6. 常见问题排查

| 现象 | 原因与处理 |
|---|---|
| 采集总是超时 | RSS 源不稳定，程序已内置超时 + 重试 2 次；可在 `config.yaml` 的 `sources` 增删源 |
| 看板整页崩溃 "no such table" | 云端没有 `data/intel.db`，程序已自动回退演示库；确认 `INTEL_DB_PATH` 指向已提交的快照 |
| 问答答非所问 | 无 embedding key 时降级为关键词检索，语义关联能力受限；配置 `EMBEDDING_API_KEY` 后恢复 |
| 邮件发不出去 | 确认 SMTP_HOST/PORT/USER/PASSWORD/TO 均已配置（收件人兼容 `SMTP_TO` 与 `MAIL_TO`）；先跑 `python run_daily.py --no-send` 验证日报生成 |
| 依赖装不上 | `requirements.txt` 里的 pandas 是看板依赖；云端漏装会启动崩，务必整份安装 |

## 7. 上线前安全检查

- 确认 `config.yaml` 中**没有**真实 API Key（只允许环境变量名）
- 确认 `.gitignore` 排除 `data/intel.db`（真实库），只提交 `intel.demo.db` 快照
- 确认 `.env` / `.pytest_cache` / `__pycache__` 未提交

