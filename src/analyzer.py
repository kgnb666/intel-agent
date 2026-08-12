"""LLM 分析模块：对未分析文章生成结构化分析（摘要/情感/标签/实体）。

设计要点：
- 走 OpenAI 兼容协议，换供应商只需改 base_url/model，代码零改动
- API key 只从环境变量读取，配置文件里只存"变量名"不存密钥
- 强制 JSON 输出 + 解析失败重试，应对大模型输出不稳定的现实问题
- 单篇失败不阻断整批，失败原因留痕，便于事后排查
"""
import json
import os
import re
import time

# JSON 输出契约：字段固定，便于入库和下游消费
_REQUIRED_FIELDS = ("summary", "sentiment", "confidence", "tags", "entities")
_SENTIMENTS = ("正面", "中性", "负面")

_PROMPT_TEMPLATE = """你是一名行业情报分析师。请分析下面这篇关于「{industry}」行业的资讯，并以 JSON 格式输出分析结果。

要求：
1. summary：一句话摘要，80 字以内，突出事实而非评论
2. sentiment：情感倾向，只能是"正面"、"中性"、"负面"之一
3. confidence：你对情感判断的置信度，0 到 1 之间的小数
4. tags：行业标签数组，1~3 个，如 ["电商", "财报"]
5. entities：文中出现的关键公司/产品名数组，没有则为空数组

只输出 JSON 对象本身，不要输出 markdown 代码块标记，不要输出任何其他文字。

标题：{title}
正文：{content}"""


class LLMConfig:
    """LLM 连接配置：环境变量优先于 config.yaml，密钥绝不落盘。"""

    def __init__(self, cfg: dict):
        llm = cfg.get("llm", {})
        self.base_url = os.environ.get("LLM_BASE_URL") or llm.get("base_url", "https://api.deepseek.com")
        self.model = os.environ.get("LLM_MODEL") or llm.get("model", "deepseek-chat")
        key_env = llm.get("api_key_env", "LLM_API_KEY")
        self.api_key = os.environ.get(key_env, "")
        self.max_per_run = int(llm.get("max_per_run", 20))
        self.temperature = float(llm.get("temperature", 0.2))

    @property
    def available(self) -> bool:
        return bool(self.api_key)


def build_prompt(industry: str, article: dict) -> str:
    """组装单篇文章的分析 prompt（dry-run 模式下也会用到它做验证）。"""
    content = article.get("summary") or article["title"]
    return _PROMPT_TEMPLATE.format(
        industry=industry, title=article["title"], content=content
    )


def _extract_json(text: str) -> dict:
    """从模型输出中提取 JSON 对象。

    模型偶尔会在 JSON 外面包裹说明文字或 ```json 代码块，
    因此先做宽容提取再严格校验字段，比直接 json.loads 更抗噪。
    """
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("输出中未找到 JSON 对象")
    data = json.loads(match.group(0))
    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"JSON 缺少字段: {missing}")
    if data["sentiment"] not in _SENTIMENTS:
        raise ValueError(f"非法情感值: {data['sentiment']}")
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    return data


class Analyzer:
    """对库内未分析文章逐篇调用 LLM，产出结构化分析结果。"""

    def __init__(self, llm_cfg: LLMConfig, industry: str, client=None, dry_run: bool = False):
        self.cfg = llm_cfg
        self.industry = industry
        # client 可注入，便于测试；dry_run 模式不创建真实客户端
        if client is None and not dry_run:
            from openai import OpenAI

            client = OpenAI(api_key=llm_cfg.api_key, base_url=llm_cfg.base_url)
        self.client = client

    def _call_with_backoff(self, prompt: str, max_retries: int = 3):
        """带指数退避的 API 调用：限流/超时是瞬态错误，退避重试成功率更高。"""
        delay = 2
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                    response_format={"type": "json_object"},
                    timeout=60,
                )
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def analyze_one(self, article: dict, dry_run: bool = False) -> dict:
        """分析单篇文章。返回 {ok, result|error, tokens, prompt}。"""
        prompt = build_prompt(self.industry, article)
        if dry_run:
            return {"ok": True, "dry_run": True, "prompt": prompt, "tokens": 0}

        last_err = None
        for _ in range(2):  # 解析失败最多重试 2 次（LLM 输出不稳定是已知风险）
            try:
                resp = self._call_with_backoff(prompt)
                result = _extract_json(resp.choices[0].message.content)
                tokens = getattr(resp.usage, "total_tokens", 0) or 0
                return {"ok": True, "result": result, "tokens": tokens}
            except Exception as e:
                last_err = e
        return {"ok": False, "error": str(last_err), "tokens": 0}

    def run(self, storage, limit: int = None, dry_run: bool = False) -> dict:
        """批量分析入口。单篇失败记录后继续，不阻断整批。"""
        limit = min(limit or self.cfg.max_per_run, self.cfg.max_per_run)
        articles = storage.unanalyzed_articles(limit)
        stats = {"total": len(articles), "success": 0, "failed": 0, "tokens": 0, "errors": []}

        for art in articles:
            out = self.analyze_one(art, dry_run=dry_run)
            if dry_run:
                stats["success"] += 1
                print(f"[DRY-RUN] 《{art['title']}》\n{out['prompt']}\n{'-' * 60}")
                continue
            if out["ok"]:
                storage.save_analysis(art["id"], out["result"], out["tokens"])
                stats["success"] += 1
                stats["tokens"] += out["tokens"]
            else:
                stats["failed"] += 1
                stats["errors"].append(f"#{art['id']} {art['title']}: {out['error']}")
        return stats
