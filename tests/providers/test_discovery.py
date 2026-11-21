"""Tests for the live model discovery and automatic selection layer."""

import json
import os

import pytest

from app.providers import (
    Candidate,
    ModelInfo,
    ProviderSpec,
    discover_models,
    fetch_models,
    rank_models,
    read_cached_models,
    select_candidates,
    write_cached_models,
)
from tests.providers.mock_provider import MockProvider, MockProviderServer


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never read or write the developer's real model cache."""
    monkeypatch.setenv("OPENMANUS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENMANUS_MODEL_CACHE_TTL", "3600")
    yield


def _groq_spec(base_url: str) -> ProviderSpec:
    return ProviderSpec(
        name="groq",
        base_url=base_url,
        api_key="test-key",
        family="groq",
    )


def _gemini_spec(base_url: str) -> ProviderSpec:
    return ProviderSpec(
        name="gemini",
        base_url=base_url,
        api_key="test-key",
        family="gemini",
    )


def _openrouter_spec(base_url: str) -> ProviderSpec:
    return ProviderSpec(
        name="openrouter",
        base_url=base_url,
        api_key="test-key",
        family="openrouter",
    )


class TestDiscovery:
    def test_groq_discovers_only_usable_chat_models(self):
        with MockProviderServer(MockProvider("groq", "groq")) as server:
            models = fetch_models(_groq_spec(server.base_url))

        ids = {model.id for model in models}
        # Non chat models are filtered out.
        assert "whisper-large-v3-turbo" not in ids
        assert "meta-llama/llama-guard-3-8b" not in ids
        # Inactive models are filtered out.
        assert "retired-model-v1" not in ids
        # Real chat models are present.
        assert "openai/gpt-oss-120b" in ids

    def test_groq_reads_pricing_and_supported_features(self):
        with MockProviderServer(MockProvider("groq", "groq")) as server:
            models = fetch_models(_groq_spec(server.base_url))

        by_id = {model.id: model for model in models}

        # Paid model -> not free.
        assert by_id["openai/gpt-oss-120b"].is_free is False
        assert by_id["openai/gpt-oss-120b"].cost == pytest.approx(0.00000075)
        assert by_id["openai/gpt-oss-120b"].max_completion_tokens == 65536

        # Zero price model -> free.
        assert by_id["qwen/qwen3-32b"].is_free is True

        # Tool support is read from supported_features.
        assert by_id["qwen/qwen3-32b"].supports_tools is True

        # Image support is read from the modalities.
        assert (
            by_id["meta-llama/llama-4-scout-17b-16e-instruct"].supports_images is True
        )
        assert by_id["qwen/qwen3-32b"].supports_images is False

    def test_gemini_strips_models_prefix(self):
        with MockProviderServer(MockProvider("gemini", "gemini")) as server:
            models = fetch_models(_gemini_spec(server.base_url))

        ids = {model.id for model in models}
        assert "gemini-2.5-flash" in ids
        assert "gemini-2.5-pro" in ids
        # Embedding and image generation models are not chat models.
        assert "gemini-embedding-2" not in ids
        assert "gemini-2.5-flash-image" not in ids

    def test_openrouter_only_keeps_zero_price_chat_models(self):
        with MockProviderServer(MockProvider("openrouter", "openrouter")) as server:
            models = fetch_models(_openrouter_spec(server.base_url))

        by_id = {model.id: model for model in models}

        assert by_id["meta-llama/llama-3.3-70b-instruct:free"].is_free is True
        assert by_id["google/gemini-2.5-flash:free"].is_free is True
        # Paid model is discovered but marked as not free.
        assert by_id["anthropic/claude-sonnet-4.5"].is_free is False
        # Batch variants and dynamic routers are dropped.
        assert "openai/gpt-4o-mini:batch" not in by_id
        assert "openrouter/auto" not in by_id
        # Tool support comes from supported_parameters.
        assert by_id["google/gemini-2.5-flash:free"].supports_tools is True
        # Image support comes from the architecture.
        assert by_id["google/gemini-2.5-flash:free"].supports_images is True
        assert by_id["meta-llama/llama-3.3-70b-instruct:free"].supports_images is False

    def test_discovery_uses_the_cache(self):
        with MockProviderServer(MockProvider("groq", "groq")) as server:
            spec = _groq_spec(server.base_url)
            first = discover_models(spec)
            assert read_cached_models("groq") is not None

            # Break the server: the cached list must still be returned.
            second = discover_models(spec)

        assert [m.id for m in first] == [m.id for m in second]

    def test_discovery_ignores_an_expired_cache(self):
        with MockProviderServer(MockProvider("groq", "groq")) as server:
            spec = _groq_spec(server.base_url)
            models = discover_models(spec)

            os.environ["OPENMANUS_MODEL_CACHE_TTL"] = "0"
            refreshed = discover_models(spec)

        assert len(refreshed) == len(models)

    def test_unreachable_provider_raises(self):
        spec = _groq_spec("http://127.0.0.1:9/v1")
        with pytest.raises(RuntimeError, match="Could not reach"):
            fetch_models(spec)

    def test_provider_without_usable_model_raises(self):
        provider = MockProvider(
            "groq",
            "groq",
            models=[
                {
                    "id": "whisper-large-v3",
                    "active": True,
                    "context_window": 448,
                }
            ],
        )
        with MockProviderServer(provider) as server:
            with pytest.raises(RuntimeError, match="usable chat model"):
                fetch_models(_groq_spec(server.base_url))


class TestFamilyInference:
    def test_the_name_wins_over_the_base_url(self):
        """A proxy or a mock server must not disable the catalogue rules."""
        from app.providers import infer_family

        assert infer_family("http://127.0.0.1:8000/v1", "groq") == "groq"
        assert infer_family("http://127.0.0.1:8000/v1", "gemini") == "gemini"
        assert infer_family("http://127.0.0.1:8000/v1", "openrouter") == "openrouter"
        assert infer_family("http://127.0.0.1:8000/v1", "my-gateway") == "generic"

    def test_the_url_is_used_when_the_name_is_unknown(self):
        from app.providers import infer_family

        assert infer_family("https://api.groq.com/openai/v1") == "groq"
        assert (
            infer_family("https://generativelanguage.googleapis.com/v1beta/openai/")
            == "gemini"
        )
        assert infer_family("https://openrouter.ai/api/v1") == "openrouter"
        assert infer_family("https://api.openai.com/v1") == "generic"

    def test_the_spec_infers_the_family_itself(self):
        spec = ProviderSpec(
            name="groq", base_url="http://127.0.0.1:8000/v1", api_key="k"
        )
        assert spec.family == "groq"


class TestRanking:
    def test_free_models_win_over_paid_ones(self):
        models = [
            ModelInfo(id="paid-big", is_free=False, cost=0.01),
            ModelInfo(id="free-small", is_free=True, cost=0.0),
        ]
        ranked = rank_models(models, free_only=True)
        assert [m.id for m in ranked] == ["free-small"]

    def test_falls_back_to_the_full_catalogue_when_nothing_is_free(self):
        models = [ModelInfo(id="paid", is_free=False, cost=0.01)]
        ranked = rank_models(models, free_only=True)
        assert [m.id for m in ranked] == ["paid"]

    def test_models_without_tool_support_are_dropped(self):
        models = [
            ModelInfo(id="no-tools", is_free=True, supports_tools=False),
            ModelInfo(id="with-tools", is_free=True, supports_tools=True),
        ]
        ranked = rank_models(models, free_only=True)
        assert [m.id for m in ranked] == ["with-tools"]

    def test_explicit_preference_wins(self):
        models = [
            ModelInfo(id="a", is_free=True),
            ModelInfo(id="b", is_free=True),
        ]
        ranked = rank_models(models, prefer=["b"], free_only=True)
        assert ranked[0].id == "b"

    def test_cheaper_model_is_preferred_within_the_same_tier(self):
        models = [
            ModelInfo(id="expensive", is_free=False, cost=0.02),
            ModelInfo(id="cheap", is_free=False, cost=0.0001),
        ]
        ranked = rank_models(models, free_only=False)
        assert ranked[0].id == "cheap"

    def test_benchmark_quality_beats_the_heuristic(self):
        models = [
            ModelInfo(id="heuristic-good", is_free=True, quality=None),
            ModelInfo(id="benchmark-good", is_free=True, quality=90.0),
        ]
        ranked = rank_models(models, free_only=True)
        assert ranked[0].id == "benchmark-good"


class TestSelectCandidates:
    def test_auto_selects_free_models_in_order(self):
        with MockProviderServer(MockProvider("groq", "groq")) as server:
            spec = _groq_spec(server.base_url)
            spec.max_candidates = 3
            candidates = select_candidates(spec)

        # The paid gpt-oss models must not be selected by default.
        assert all("gpt-oss" not in c.model for c in candidates)
        assert candidates
        assert isinstance(candidates[0], Candidate)

    def test_explicit_model_bypasses_discovery(self):
        spec = ProviderSpec(
            name="groq",
            base_url="http://127.0.0.1:9/v1",
            api_key="k",
            family="groq",
            model="llama-3.3-70b-versatile",
        )
        candidates = select_candidates(spec)
        assert len(candidates) == 1
        assert candidates[0].model == "llama-3.3-70b-versatile"

    def test_max_completion_tokens_is_known_for_the_first_candidate(self):
        with MockProviderServer(MockProvider("groq", "groq")) as server:
            spec = _groq_spec(server.base_url)
            spec.model = "auto"
            candidates = select_candidates(spec)

        assert candidates[0].max_completion_tokens


class TestCache:
    def test_roundtrip(self, tmp_path):
        models = [ModelInfo(id="m1", is_free=True, supports_tools=True)]
        write_cached_models("unit-test", models)
        cached = read_cached_models("unit-test")
        assert cached is not None
        assert cached[0].id == "m1"
        assert cached[0].is_free is True

    def test_cache_file_is_json(self, tmp_path):
        models = [ModelInfo(id="m1")]
        write_cached_models("unit-test-json", models)
        path = tmp_path / "models" / "unit-test-json.json"
        payload = json.loads(path.read_text())
        assert payload["models"][0]["id"] == "m1"
