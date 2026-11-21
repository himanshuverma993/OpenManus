"""Tests for the automatic provider chain resolution in :mod:`app.config`."""

import textwrap

import pytest

from app.config import Config, ProviderSettings, expand_env_vars


def write_config(tmp_path, content: str):
    path = tmp_path / "config.toml"
    path.write_text(textwrap.dedent(content))
    return path


def load_config(tmp_path, monkeypatch, content: str, env=None):
    path = write_config(tmp_path, content)
    monkeypatch.setattr(Config, "_get_config_path", staticmethod(lambda: path))
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    instance = Config()
    instance._initialized = False
    instance._config = None
    instance._load_initial_config()
    return instance


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENMANUS_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class TestEnvExpansion:
    def test_expands_plain_placeholders(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        assert expand_env_vars("${MY_KEY}") == "secret"

    def test_expands_with_default(self, monkeypatch):
        monkeypatch.delenv("MY_KEY", raising=False)
        assert expand_env_vars("${MY_KEY:-fallback}") == "fallback"

    def test_expands_nested_structures(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        assert expand_env_vars({"a": ["${MY_KEY}"], "b": {"c": "${MY_KEY}"}}) == {
            "a": ["secret"],
            "b": {"c": "secret"},
        }


class TestAutomaticMode:
    def test_model_auto_uses_every_builtin_provider(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "auto"
            max_tokens = 4096
            """,
            env={"GROQ_API_KEY": "groq-key"},
        )
        chain = instance.llm["default"].providers
        names = [provider.name for provider in chain]
        assert names == ["groq", "gemini", "openrouter"]
        assert chain[0].api_key == "groq-key"
        # Providers without a key stay in the chain but are skipped later.
        assert chain[1].api_key == ""

    def test_explicit_provider_list_defines_the_order(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "auto"
            providers = ["openrouter", "groq"]

            [providers.groq]
            api_key = "g-key"

            [providers.openrouter]
            api_key = "or-key"
            """,
        )
        chain = instance.llm["default"].providers
        assert [provider.name for provider in chain] == ["openrouter", "groq"]
        assert chain[0].api_key == "or-key"

    def test_provider_defaults_are_used(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "auto"
            providers = ["groq", "gemini", "openrouter"]

            [providers.groq]
            api_key = "g-key"

            [providers.gemini]
            api_key = "gem-key"

            [providers.openrouter]
            api_key = "or-key"
            """,
        )
        chain = {
            provider.name: provider for provider in instance.llm["default"].providers
        }
        assert chain["groq"].base_url == "https://api.groq.com/openai/v1"
        assert chain["gemini"].base_url.startswith(
            "https://generativelanguage.googleapis.com/v1beta/openai"
        )
        assert chain["openrouter"].base_url == "https://openrouter.ai/api/v1"
        # OpenRouter asks for attribution headers.
        assert chain["openrouter"].extra_headers["X-Title"] == "OpenManus"

    def test_environment_variables_are_used_for_the_keys(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "auto"
            providers = ["groq", "gemini", "openrouter"]
            """,
            env={
                "GROQ_API_KEY": "env-groq",
                "GEMINI_API_KEY": "env-gemini",
                "OPENROUTER_API_KEY": "env-openrouter",
            },
        )
        chain = {
            provider.name: provider for provider in instance.llm["default"].providers
        }
        assert chain["groq"].api_key == "env-groq"
        assert chain["gemini"].api_key == "env-gemini"
        assert chain["openrouter"].api_key == "env-openrouter"

    def test_google_api_key_is_accepted_for_gemini(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "auto"
            providers = ["gemini"]
            """,
            env={"GOOGLE_API_KEY": "google-key"},
        )
        assert instance.llm["default"].providers[0].api_key == "google-key"

    def test_legacy_single_model_still_works(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "gpt-4o"
            base_url = "https://api.openai.com/v1"
            api_key = "sk-test"
            """,
        )
        chain = instance.llm["default"].providers
        assert len(chain) == 1
        assert chain[0].model == "gpt-4o"
        assert chain[0].base_url == "https://api.openai.com/v1"

    def test_env_placeholder_in_the_config_file(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "auto"
            providers = ["groq"]

            [providers.groq]
            api_key = "${MY_GROQ_KEY}"
            """,
            env={"MY_GROQ_KEY": "from-env"},
        )
        assert instance.llm["default"].providers[0].api_key == "from-env"

    def test_provider_preferences_are_forwarded(self, tmp_path, monkeypatch):
        instance = load_config(
            tmp_path,
            monkeypatch,
            """
            [llm]
            model = "auto"
            providers = ["groq"]

            [providers.groq]
            api_key = "g-key"
            prefer = ["llama-3.3-70b-versatile"]
            max_candidates = 5
            free_only = false
            """,
        )
        provider = instance.llm["default"].providers[0]
        assert provider.prefer == ["llama-3.3-70b-versatile"]
        assert provider.max_candidates == 5
        assert provider.free_only is False


class TestProviderSettings:
    def test_defaults(self):
        provider = ProviderSettings(name="groq")
        assert provider.model == "auto"
        assert provider.free_only is True
        assert provider.max_candidates == 3
        assert provider.prefer == []
