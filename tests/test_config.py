"""config 模块单元测试：字段级校验、错误消息定位、入口友好报错。"""
import os

import pytest
import yaml

from src.config import ConfigError, load_config, load_config_or_exit, save_keywords


def _cfg():
    return {
        "industry": {"name": "电商与消费", "keywords": ["电商", "消费"]},
        "sources": [{"name": "源A", "url": "https://a.example.com/feed"}],
        "storage": {"db_path": "data/intel.db"},
        "crawl": {"max_items_per_source": 50, "title_similarity_threshold": 0.85},
        "llm": {"base_url": "https://api.deepseek.com", "model": "deepseek-chat",
                "api_key_env": "LLM_API_KEY", "max_per_run": 20},
        "embedding": {"base_url": "https://api.siliconflow.cn/v1", "model": "BAAI/bge-m3",
                      "api_key_env": "EMBEDDING_API_KEY", "similarity_threshold": 0.6,
                      "top_k": 3},
    }


def _write(tmp_path, cfg) -> str:
    path = tmp_path / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return str(path)


class TestLoadConfig:
    def test_valid_config_loads(self, tmp_path):
        path = _write(tmp_path, _cfg())
        cfg = load_config(path)
        assert cfg["industry"]["name"] == "电商与消费"
        assert cfg["sources"][0]["url"].startswith("https://")
        assert os.path.isabs(cfg["storage"]["db_path"])
        assert cfg["storage"]["db_path"].endswith(os.path.join("data", "intel.db"))

    def test_intel_db_path_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("INTEL_DB_PATH", "other.db")
        cfg = load_config(_write(tmp_path, _cfg()))
        assert cfg["storage"]["db_path"].endswith("other.db")

    def test_missing_section_raises_with_section_name(self, tmp_path):
        cfg = _cfg()
        cfg.pop("sources")
        with pytest.raises(ConfigError, match="sources 段缺失") as e:
            load_config(_write(tmp_path, cfg))
        assert e.value.section == "sources"

    def test_top_level_not_mapping(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="顶层"):
            load_config(str(path))


class TestIndustryValidation:
    def test_empty_keywords(self, tmp_path):
        cfg = _cfg()
        cfg["industry"]["keywords"] = []
        with pytest.raises(ConfigError, match="industry.keywords") as e:
            load_config(_write(tmp_path, cfg))
        assert e.value.section == "industry"

    def test_blank_keyword_item(self, tmp_path):
        cfg = _cfg()
        cfg["industry"]["keywords"] = ["电商", "  "]
        with pytest.raises(ConfigError, match="industry.keywords"):
            load_config(_write(tmp_path, cfg))

    def test_blank_industry_name(self, tmp_path):
        cfg = _cfg()
        cfg["industry"]["name"] = "  "
        with pytest.raises(ConfigError, match="industry.name"):
            load_config(_write(tmp_path, cfg))


class TestSourcesValidation:
    def test_empty_sources(self, tmp_path):
        cfg = _cfg()
        cfg["sources"] = []
        with pytest.raises(ConfigError, match="至少一个数据源") as e:
            load_config(_write(tmp_path, cfg))
        assert e.value.section == "sources"

    def test_source_missing_url(self, tmp_path):
        cfg = _cfg()
        cfg["sources"] = [{"name": "无URL源"}]
        with pytest.raises(ConfigError, match=r"sources\[0\]\.url"):
            load_config(_write(tmp_path, cfg))


class TestCrawlValidation:
    @pytest.mark.parametrize("value", [0, -3, "50", 2.5, True])
    def test_max_items_must_be_positive_int(self, tmp_path, value):
        cfg = _cfg()
        cfg["crawl"]["max_items_per_source"] = value
        with pytest.raises(ConfigError, match="crawl.max_items_per_source") as e:
            load_config(_write(tmp_path, cfg))
        assert e.value.section == "crawl"


class TestLlmEmbeddingValidation:
    def test_invalid_llm_api_key_env(self, tmp_path):
        cfg = _cfg()
        cfg["llm"]["api_key_env"] = "123 BAD"
        with pytest.raises(ConfigError, match="llm.api_key_env") as e:
            load_config(_write(tmp_path, cfg))
        assert e.value.section == "llm"

    @pytest.mark.parametrize("value", [1.5, -0.1, "0.6", True])
    def test_embedding_threshold_out_of_range(self, tmp_path, value):
        cfg = _cfg()
        cfg["embedding"]["similarity_threshold"] = value
        with pytest.raises(ConfigError, match="embedding.similarity_threshold") as e:
            load_config(_write(tmp_path, cfg))
        assert e.value.section == "embedding"

    def test_embedding_top_k_invalid(self, tmp_path):
        cfg = _cfg()
        cfg["embedding"]["top_k"] = 0
        with pytest.raises(ConfigError, match="embedding.top_k"):
            load_config(_write(tmp_path, cfg))


class TestAlertValidation:
    def test_valid_alert_section(self, tmp_path):
        cfg = _cfg()
        cfg["alert"] = {"webhook": "", "min_confidence": 0.8}
        assert load_config(_write(tmp_path, cfg))["alert"]["min_confidence"] == 0.8

    def test_invalid_alert_min_confidence(self, tmp_path):
        cfg = _cfg()
        cfg["alert"] = {"webhook": "", "min_confidence": 1.5}
        with pytest.raises(ConfigError, match="alert.min_confidence") as e:
            load_config(_write(tmp_path, cfg))
        assert e.value.section == "alert"

    def test_invalid_alert_webhook_type(self, tmp_path):
        cfg = _cfg()
        cfg["alert"] = {"webhook": 123}
        with pytest.raises(ConfigError, match="alert.webhook"):
            load_config(_write(tmp_path, cfg))


class TestFileErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="找不到配置文件") as e:
            load_config(str(tmp_path / "nope.yaml"))
        assert e.value.section == "文件"

    def test_malformed_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("industry: [unclosed\n  bad: : :\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="解析失败") as e:
            load_config(str(path))
        assert e.value.section == "文件"


class TestLoadConfigOrExit:
    def test_broken_config_prints_friendly_hint_and_exits(self, tmp_path, capsys):
        cfg = _cfg()
        cfg["industry"]["keywords"] = []
        path = _write(tmp_path, cfg)
        with pytest.raises(SystemExit) as exc:
            load_config_or_exit(path)
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert "[配置错误]" in err
        assert "industry.keywords" in err
        assert "请检查 config.yaml 的「industry」段后重试。" in err

    def test_valid_config_returns(self, tmp_path):
        cfg = load_config_or_exit(_write(tmp_path, _cfg()))
        assert cfg["industry"]["name"] == "电商与消费"


class TestSaveKeywords:
    def test_save_keywords_preserves_comments_and_relative_db_path(self, tmp_path):
        raw_yaml = """# 顶层头部注释
# 重要的行业情报设计要点
industry:
  name: "电商与消费"
  keywords:
    - 电商
    - 消费

# 数据源注释说明
sources:
  - name: "源A"
    url: "https://a.example.com/feed"

storage:
  db_path: "data/intel.db"

crawl:
  max_items_per_source: 50
  title_similarity_threshold: 0.85

llm:
  base_url: "https://api.deepseek.com"
  model: "deepseek-chat"
  api_key_env: "LLM_API_KEY"
  max_per_run: 20

embedding:
  base_url: "https://api.siliconflow.cn/v1"
  model: "BAAI/bge-m3"
  api_key_env: "EMBEDDING_API_KEY"
  similarity_threshold: 0.6
  top_k: 3
"""
        path = tmp_path / "config.yaml"
        path.write_text(raw_yaml, encoding="utf-8")

        new_kws = ["新消费", "直播带货", "拼多多"]
        save_keywords(new_kws, str(path))

        # 验证文本中依然完整保留中文注释
        saved_text = path.read_text(encoding="utf-8")
        assert "# 顶层头部注释" in saved_text
        assert "# 重要的行业情报设计要点" in saved_text
        assert "# 数据源注释说明" in saved_text
        assert 'db_path: "data/intel.db"' in saved_text

        # 验证解析后的内容
        loaded = load_config(str(path))
        assert loaded["industry"]["keywords"] == new_kws
        assert loaded["storage"]["db_path"].endswith(os.path.join("data", "intel.db"))

    def test_save_empty_keywords_raises(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("industry:\n  keywords:\n    - a\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="industry.keywords"):
            save_keywords([], str(path))
        with pytest.raises(ConfigError, match="industry.keywords"):
            save_keywords(["   "], str(path))

    def test_save_keywords_special_characters(self, tmp_path):
        cfg = _cfg()
        path = _write(tmp_path, cfg)
        save_keywords(["AI:智能", "电商#零售", "[快讯]"], path)
        loaded = load_config(path)
        assert loaded["industry"]["keywords"] == ["AI:智能", "电商#零售", "[快讯]"]

