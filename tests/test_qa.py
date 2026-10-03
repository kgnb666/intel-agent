"""qa 模块单元测试：中文分词、字段加权排序、混合检索、回答组装与降级路径。

覆盖的重点是"线上真实踩过的坑"：
- 中文问句不能按空格切词，否则 "2026国庆电影票房" 整串 LIKE 必然零命中；
- 零命中时要给出可操作建议，而不是一句"未检索到"；
- 未配置 LLM 时也必须产出可读的要点式答案；
- 向量不可用/维度不符时要降级到关键词检索而不是抛错。
"""
import json
import sqlite3

import numpy as np

from src.qa import (
    CallBudget,
    Hit,
    answer_question,
    build_extractive_answer,
    build_llm_messages,
    build_no_hit_answer,
    citation_list,
    expand_tokens,
    hit_stats,
    hybrid_merge,
    keyword_rank,
    load_candidates,
    normalize_date,
    score_row,
    snippet,
    suggest_topics,
    tokenize,
    vector_rank,
)


# --------------------------------------------------------------------------
# 分词
# --------------------------------------------------------------------------


class TestTokenize:
    def test_chinese_question_is_split_into_ngrams(self):
        tokens = tokenize("2026国庆电影票房")
        assert "2026" in tokens
        for expected in ("国庆", "电影", "票房"):
            assert expected in tokens

    def test_short_chinese_word_kept_whole(self):
        assert "京东" in tokenize("京东最近有什么动作")

    def test_stopwords_removed(self):
        tokens = tokenize("这周拼多多有什么动态？")
        assert "拼多" in tokens or "拼多多" in tokens
        for stop in ("这周", "什么", "动态"):
            assert stop not in tokens

    def test_generic_domain_words_are_stopwords(self):
        # "行业"在这个项目里几乎出现在所有语料中，留着只会把无关文章拉进结果
        tokens = tokenize("最近电影行业有什么值得关注的动态？")
        assert "行业" not in tokens
        assert "电影" in tokens

    def test_ascii_tokens_lowercased(self):
        tokens = tokenize("AI 手机 iPhone17 发布")
        assert "ai" in tokens
        assert "iphone17" in tokens

    def test_tokens_are_deduplicated(self):
        tokens = tokenize("拼多多 拼多多")
        assert len(tokens) == len(set(tokens))

    def test_empty_input(self):
        assert tokenize("") == []
        assert tokenize(None) == []

    def test_expand_tokens_adds_alias(self):
        expanded = expand_tokens(["双11"])
        assert "双十一" in expanded
        assert expanded[0] == "双11"  # 原词权重更高，排在前面


# --------------------------------------------------------------------------
# 排序与召回
# --------------------------------------------------------------------------


def _row(**kw):
    row = {
        "id": 1,
        "title": "",
        "url": "https://example.com/1",
        "source": "测试源",
        "published": "2026-09-10 08:00:00",
        "fetched_at": "2026-09-10 09:00:00",
        "summary": "",
        "summary_ai": "",
        "sentiment": "中性",
        "tags": None,
        "entities": None,
        "embedding": None,
    }
    row.update(kw)
    return row


class TestKeywordRank:
    def test_matching_is_case_insensitive(self):
        # 查询词会小写化，文档侧也必须小写，否则 "iPhone" 标题匹配不到 "iphone" 查询
        score, matched = score_row(tokenize("iphone"), _row(title="支持 iPhone Duo 快充"))
        assert score > 0
        assert "iphone" in matched

    def test_english_query_matches_uppercase_title(self):
        hits = keyword_rank([_row(id=1, title="iPhone 17 发布")], tokenize("iPhone"), limit=5)
        assert [h.id for h in hits] == [1]

    def test_title_match_beats_summary_match(self):
        tokens = tokenize("拼多多")
        title_score, _ = score_row(tokens, _row(title="拼多多发布财报"))
        summary_score, _ = score_row(tokens, _row(summary="拼多多发布财报"))
        assert title_score > summary_score

    def test_match_in_tags_counts(self):
        tokens = tokenize("财报")
        tagged, _ = score_row(tokens, _row(tags='["电商", "财报"]'))
        assert tagged > 0

    def test_ranking_order_and_score_normalisation(self):
        rows = [
            _row(id=1, title="拼多多发布新财报"),
            _row(id=2, summary="有分析提到拼多多"),
            _row(id=3, title="完全无关的新闻"),
        ]
        hits = keyword_rank(rows, tokenize("拼多多财报"), limit=5)
        assert [h.id for h in hits] == [1, 2]
        assert hits[0].score == 1.0
        assert all(0 <= h.score <= 1 for h in hits)

    def test_single_char_only_match_is_not_a_hit(self):
        rows = [_row(id=1, title="电")]
        assert keyword_rank(rows, tokenize("电商"), limit=5) == []

    def test_relaxed_fallback_marks_weak_hits(self):
        # 两个单字碎片都命中：严格匹配不通过，宽松兜底给出降权的弱匹配
        rows = [_row(id=1, title="电影票房")]
        hits = keyword_rank(rows, ["电", "影"], limit=5)
        assert len(hits) == 1
        assert hits[0].score <= 0.6

    def test_no_match_returns_empty(self):
        rows = [_row(id=1, title="新能源汽车销量")]
        assert keyword_rank(rows, tokenize("白酒行业"), limit=5) == []

    def test_year_only_match_is_not_a_hit(self):
        # "2026国庆电影票房" 里只有年份能命中全库文章，不能当成有效命中
        rows = [_row(id=1, title="2026 胡润中国品牌榜发布")]
        assert keyword_rank(rows, tokenize("2026国庆电影票房"), limit=5) == []

    def test_meaningful_token_beside_year_still_hits(self):
        rows = [_row(id=1, title="2026 折叠屏手机盘点")]
        assert len(keyword_rank(rows, tokenize("2026折叠屏手机"), limit=5)) == 1

    def test_multi_term_match_outranks_single_common_term(self):
        rows = [
            _row(id=1, title="折叠屏手机新品曝光"),
            _row(id=2, title="手机散热器上架京东"),
        ]
        hits = keyword_rank(rows, tokenize("折叠屏手机有哪些新品？"), limit=5)
        assert hits[0].id == 1

    def test_single_common_term_match_still_returned(self):
        # 「最近电影行业有什么动态？」在只有一篇电影报道的库里必须仍能命中：
        # 为了压掉噪声而要求"命中≥2个词"会误伤这种合法结果
        rows = [_row(id=1, title="我国电影银幕总数接近 9.6 万块"), _row(id=2, title="手机销量下滑")]
        hits = keyword_rank(rows, tokenize("最近电影行业有什么值得关注的动态？"), limit=5)
        assert [h.id for h in hits] == [1]

    def test_short_query_matches_common_word(self):
        rows = [_row(id=i, title=f"某厂商发布手机新品 {i}") for i in range(1, 9)]
        assert len(keyword_rank(rows, tokenize("手机"), limit=5)) == 5


class TestVectorRank:
    def test_orders_by_cosine_similarity(self):
        rows = [
            _row(id=1, embedding=np.array([1.0, 0.0], dtype=np.float32).tobytes()),
            _row(id=2, embedding=np.array([0.0, 1.0], dtype=np.float32).tobytes()),
        ]
        hits = vector_rank(rows, np.array([0.9, 0.1], dtype=np.float32), limit=5)
        assert [h.id for h in hits] == [1, 2]
        assert hits[0].similarity > hits[1].similarity

    def test_dimension_mismatch_returns_empty(self):
        rows = [_row(id=1, embedding=np.array([1.0, 0.0], dtype=np.float32).tobytes())]
        assert vector_rank(rows, np.array([1.0, 0.0, 0.0], dtype=np.float32)) == []

    def test_rows_without_embedding_are_skipped(self):
        rows = [_row(id=1), _row(id=2)]
        assert vector_rank(rows, np.array([1.0], dtype=np.float32)) == []


class TestHybridMerge:
    @staticmethod
    def _kw(**kw):
        hit = Hit(id=kw["id"], title=kw.get("title", "标题"), url="")
        hit.raw_score = kw.get("raw", 1.0)
        hit.score = kw.get("raw", 1.0)
        return hit

    @staticmethod
    def _vec(**kw):
        hit = Hit(id=kw["id"], title=kw.get("title", "标题"), url="")
        hit.similarity = kw.get("sim", 0.8)
        hit.score = hit.similarity
        return hit

    def test_item_hit_by_both_retrievers_ranks_first(self):
        kw = [self._kw(id=1, raw=1.0), self._kw(id=2, raw=0.5)]
        vec = [self._vec(id=2, sim=0.95), self._vec(id=3, sim=0.4)]
        merged = hybrid_merge(kw, vec, limit=3)
        assert merged[0].id == 2
        assert {h.id for h in merged} == {1, 2, 3}
        assert all(0 <= h.score <= 1 for h in merged)

    def test_vector_only_input_returns_vector_hits(self):
        vec = [self._vec(id=7, sim=0.9)]
        assert [h.id for h in hybrid_merge([], vec, limit=3)] == [7]

    def test_keyword_only_input_returns_keyword_hits(self):
        kw = [self._kw(id=5, raw=2.0)]
        assert [h.id for h in hybrid_merge(kw, [], limit=3)] == [5]


# --------------------------------------------------------------------------
# 概览与回答组装
# --------------------------------------------------------------------------


class TestCallBudget:
    def test_allows_up_to_limit_then_blocks(self):
        budget = CallBudget(max_per_window=2, window_seconds=100, clock=lambda: 0.0)
        assert budget.allow() is True
        assert budget.allow() is True
        assert budget.allow() is False
        assert budget.remaining() == 0
        assert budget.used == 2

    def test_window_slides_over_time(self):
        now = {"t": 1000.0}
        budget = CallBudget(max_per_window=1, window_seconds=60, clock=lambda: now["t"])
        assert budget.allow() is True
        assert budget.allow() is False
        now["t"] += 61
        assert budget.allow() is True
        assert budget.used == 1

    def test_remaining_counts_down(self):
        budget = CallBudget(max_per_window=3, window_seconds=60, clock=lambda: 0.0)
        budget.allow()
        assert budget.remaining() == 2


class TestStatsAndSnippet:
    def test_normalize_date_handles_rfc822_and_iso(self):
        assert normalize_date("Wed, 12 Aug 2026 10:45:21 GMT") == "2026-08-12"
        assert normalize_date("2026-09-10T18:51:53") == "2026-09-10"
        assert normalize_date("2026-09-10 08:00:00") == "2026-09-10"
        assert normalize_date("") == ""
        assert normalize_date("不是日期") == ""

    def test_hit_date_uses_normalised_published(self):
        hit = Hit(id=1, title="A", published="Wed, 12 Aug 2026 10:45:21 GMT",
                  fetched_at="2026-08-12T18:51:53")
        assert hit.date == "2026-08-12"

    def test_hit_stats_summarises_sources_and_dates(self):
        hits = [
            Hit(id=1, title="A", source="36氪", fetched_at="2026-09-01 10:00:00",
                sentiment="正面", entities=["拼多多"], score=1.0),
            Hit(id=2, title="B", source="36氪", fetched_at="2026-09-08 10:00:00",
                sentiment="负面", entities=["拼多多", "淘宝"], score=0.8),
        ]
        stats = hit_stats(hits, scanned=59)
        assert stats["hits"] == 2
        assert stats["scanned"] == 59
        assert stats["date_from"] == "2026-09-01"
        assert stats["date_to"] == "2026-09-08"
        assert stats["sources"][0] == ("36氪", 2)
        assert stats["sentiments"] == {"正面": 1, "负面": 1}
        assert stats["entities"][0] == ("拼多多", 2)

    def test_snippet_centers_on_matched_term(self):
        text = "前" * 200 + "拼多多" + "后" * 200
        out = snippet(text, ["拼多多"], width=60)
        assert "拼多多" in out
        assert out.startswith("…") and out.endswith("…")
        assert len(out) <= 62

    def test_snippet_without_match_truncates(self):
        assert snippet("短文本", ["不存在"]) == "短文本"
        assert len(snippet("字" * 300, ["不存在"], width=100)) <= 101


class TestAnswerAssembly:
    @staticmethod
    def _hits():
        return [
            Hit(id=1, title="拼多多发布财报 [独家]", url="https://a/1", source="36氪",
                published="2026-09-09 10:00:00", summary="拼多多二季度营收增长。",
                sentiment="正面", score=1.0, matched=["拼多多", "财报"]),
            Hit(id=2, title="淘宝跟进补贴", url="https://a/2", source="IT之家",
                published="2026-09-08 10:00:00", summary="平台补贴战升级。",
                sentiment="中性", score=0.7, matched=["补贴"]),
        ]

    def test_extractive_answer_contains_points_and_links(self):
        hits = self._hits()
        answer = build_extractive_answer("拼多多财报", hits, hit_stats(hits, scanned=59), "关键词")
        assert "库内命中 2 条" in answer
        assert "拼多多发布财报 【独家】" in answer  # 方括号转义，避免被当作 Markdown 链接
        assert "https://a/1" in answer
        assert "抽取式回答" in answer
        assert "36氪" in answer

    def test_no_hit_answer_gives_next_steps(self):
        answer = build_no_hit_answer("2026国庆电影票房", {"scanned": 35}, [("电商", 12), ("消费", 9)])
        assert "未检索到" in answer
        assert "35 篇" in answer
        assert "`电商` 12" in answer
        assert "`电商`" in answer.split("**可以试试**")[1]  # 改写建议用的是库内真实话题
        assert "立即采集" in answer

    def test_no_hit_answer_without_topics_still_actionable(self):
        answer = build_no_hit_answer("冷门问题", {"scanned": 0}, [])
        assert "可以试试" in answer
        assert "立即采集" in answer

    def test_llm_messages_number_the_sources(self):
        hits = self._hits()
        messages = build_llm_messages(
            "拼多多财报如何？", hits, history=[{"role": "user", "content": "上一个问题"}]
        )
        assert messages[0]["role"] == "system"
        assert "编号" in messages[0]["content"]
        user_msg = messages[-1]["content"]
        assert "[1]" in user_msg and "[2]" in user_msg
        assert any(m["content"] == "上一个问题" for m in messages)

    def test_citation_list_is_numbered_like_the_answer(self):
        refs = citation_list(self._hits())
        assert refs.splitlines()[0].startswith("1. [拼多多发布财报 【独家】]")
        assert refs.splitlines()[1].startswith("2. [淘宝跟进补贴]")


# --------------------------------------------------------------------------
# 端到端（SQLite）
# --------------------------------------------------------------------------


def _make_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "qa.db"))
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY, title TEXT, url TEXT, source TEXT,
            published TEXT, fetched_at TEXT, summary TEXT, embedding BLOB
        );
        CREATE TABLE analysis (
            article_id INTEGER PRIMARY KEY, summary_ai TEXT, sentiment TEXT,
            tags TEXT, entities TEXT
        );
        """
    )
    rows = [
        (1, "拼多多二季度财报超预期", "https://a/1", "36氪", "2026-09-09",
         "2026-09-09 10:00:00", "拼多多营收同比增长。", ["电商", "财报"], ["拼多多"]),
        (2, "平台补贴战升级", "https://a/2", "IT之家", "2026-09-08",
         "2026-09-08 10:00:00", "电商平台补贴力度加大。", ["电商"], ["淘宝"]),
        (3, "新款折叠屏手机发布", "https://a/3", "IT之家", "2026-09-07",
         "2026-09-07 10:00:00", "消费电子新品。", ["手机"], ["小米"]),
    ]
    for aid, title, url, source, published, fetched_at, summary, tags, entities in rows:
        conn.execute(
            "INSERT INTO articles (id, title, url, source, published, fetched_at, summary)"
            " VALUES (?,?,?,?,?,?,?)",
            (aid, title, url, source, published, fetched_at, summary),
        )
        conn.execute(
            "INSERT INTO analysis (article_id, summary_ai, sentiment, tags, entities)"
            " VALUES (?,?,?,?,?)",
            (
                aid,
                summary,
                "中性",
                json.dumps(tags, ensure_ascii=False),
                json.dumps(entities, ensure_ascii=False),
            ),
        )
    conn.commit()
    return conn


class _FakeLLMConfig:
    available = True
    model = "fake-model"
    api_key = "sk-test"
    base_url = "https://example.invalid"


class _EchoClient:
    """模拟一次成功的 LLM 调用，用于验证引用编号拼接。"""

    class chat:  # noqa: N801 - 对齐 OpenAI SDK 的嵌套结构
        class completions:  # noqa: N801
            @staticmethod
            def create(**_kwargs):
                message = type("Msg", (), {"content": "结论：拼多多二季度营收增长 [1]。"})()
                choice = type("Choice", (), {"message": message})()
                return type("Resp", (), {"choices": [choice]})()


class _ExplodingClient:
    """模拟 LLM 调用异常，验证降级路径。"""

    class chat:  # noqa: N801 - 对齐 OpenAI SDK 的嵌套结构
        class completions:  # noqa: N801
            @staticmethod
            def create(**_kwargs):
                raise RuntimeError("boom")


class TestAnswerQuestion:
    def test_llm_answer_contains_numbered_citations(self, tmp_path):
        conn = _make_db(tmp_path)
        try:
            result = answer_question(
                conn, "拼多多财报", llm_cfg=_FakeLLMConfig(), llm_client=_EchoClient()
            )
        finally:
            conn.close()
        assert result.llm_used is True
        assert "[1]" in result.answer
        assert "**引用来源**" in result.answer
        assert "1. [" in result.answer

    def test_exhausted_budget_falls_back_to_extractive(self, tmp_path):
        conn = _make_db(tmp_path)
        budget = CallBudget(max_per_window=1, window_seconds=3600, clock=lambda: 1000.0)
        assert budget.allow() is True  # 先用掉唯一一次额度
        try:
            result = answer_question(
                conn, "拼多多财报", llm_cfg=_FakeLLMConfig(),
                llm_client=_EchoClient(), budget=budget,
            )
        finally:
            conn.close()
        assert result.llm_used is False
        assert "频率上限" in result.warning
        assert "抽取式回答" in result.answer

    def test_keyword_mode_returns_hits_without_llm(self, tmp_path):
        conn = _make_db(tmp_path)
        try:
            result = answer_question(conn, "拼多多财报怎么样？")
        finally:
            conn.close()
        assert result.mode == "keyword"
        assert result.llm_used is False
        assert result.hit_count >= 1
        assert result.hits[0].id == 1
        assert "库内命中" in result.answer
        assert result.stats["scanned"] == 3

    def test_no_hit_mode_suggests_topics(self, tmp_path):
        conn = _make_db(tmp_path)
        try:
            result = answer_question(conn, "2026国庆电影票房")
        finally:
            conn.close()
        assert result.mode == "none"
        assert result.hit_count == 0
        assert "未检索到" in result.answer
        assert result.suggestions

    def test_question_without_keywords_asks_for_one(self, tmp_path):
        conn = _make_db(tmp_path)
        try:
            result = answer_question(conn, "？?")
        finally:
            conn.close()
        assert result.mode == "none"
        assert "没有可用于检索的关键词" in result.answer

    def test_days_filter_limits_scan(self, tmp_path):
        conn = _make_db(tmp_path)
        try:
            candidates = load_candidates(conn, days=1)
        finally:
            conn.close()
        assert candidates == []  # 库内都是历史数据，最近一天没有内容

    def test_llm_failure_degrades_to_extractive_answer(self, tmp_path):
        conn = _make_db(tmp_path)
        try:
            result = answer_question(
                conn, "拼多多财报", llm_cfg=_FakeLLMConfig(), llm_client=_ExplodingClient()
            )
        finally:
            conn.close()
        assert result.llm_used is False
        assert "大模型生成失败" in result.warning
        assert "抽取式回答" in result.answer

    def test_suggest_topics_counts_tags_and_entities(self, tmp_path):
        conn = _make_db(tmp_path)
        try:
            topics = dict(suggest_topics(conn, limit=10))
        finally:
            conn.close()
        assert topics["电商"] == 2
        assert topics["拼多多"] == 1
