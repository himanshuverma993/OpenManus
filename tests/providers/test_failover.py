"""End-to-end tests for the automatic provider/model failover."""

import asyncio

import pytest

from app.config import LLMSettings, ProviderSettings
from app.exceptions import TokenLimitExceeded
from app.llm import LLM
from app.schema import Message
from tests.providers.mock_provider import MockProvider, MockProviderServer


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENMANUS_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("OPENMANUS_MODEL_CACHE_TTL", "3600")
    yield


@pytest.fixture(autouse=True)
def clean_llm_instances():
    """The LLM class caches instances per configuration name."""
    LLM._instances.clear()
    yield
    LLM._instances.clear()


def build_llm(providers, name="default", **overrides):
    payload = {
        "model": "auto",
        "base_url": "",
        "api_key": "",
        "max_tokens": 1024,
        "temperature": 1.0,
        "api_type": "",
        "api_version": "",
        "providers": providers,
    }
    payload.update(overrides)
    settings = LLMSettings(**payload)
    return LLM(name, llm_config={name: settings})


def provider_settings(name, base_url, family, **overrides):
    return ProviderSettings(
        name=name,
        base_url=base_url,
        api_key="test-key",
        family=family,
        model="auto",
        max_candidates=3,
        **overrides,
    )


class TestSingleProvider:
    @pytest.mark.asyncio
    async def test_groq_tool_call_succeeds_without_message_name(self):
        """Groq rejects ``messages[].name`` - the client must not send it."""
        provider = MockProvider("groq", "groq", reply="tool answer")
        with MockProviderServer(provider) as server:
            llm = build_llm([provider_settings("groq", server.base_url, "groq")])
            response = await llm.ask_tool(
                messages=[
                    Message.user_message("hi"),
                    Message.tool_message(
                        content="result", name="python_execute", tool_call_id="call_1"
                    ),
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "python_execute",
                            "description": "run python",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                            },
                        },
                    }
                ],
            )

        assert response is not None
        # The request must not contain the forbidden field.
        assert provider.requests, "the provider did not receive a request"
        for message in provider.requests[0]["messages"]:
            assert "name" not in message

    @pytest.mark.asyncio
    async def test_max_tokens_are_clamped_to_the_model_limit(self):
        provider = MockProvider("groq", "groq", reply="ok")
        with MockProviderServer(provider) as server:
            llm = build_llm(
                [provider_settings("groq", server.base_url, "groq")],
                max_tokens=999999,
            )
            await llm.ask(messages=[Message.user_message("hi")], stream=False)

        request = provider.requests[0]
        limit = request.get("max_tokens") or request.get("max_completion_tokens")
        assert limit is not None
        assert limit <= 65536  # gpt-oss-120b / qwen limit from the catalogue

    @pytest.mark.asyncio
    async def test_streaming_request_is_consumed(self):
        provider = MockProvider("groq", "groq", reply="streamed answer")
        with MockProviderServer(provider) as server:
            llm = build_llm([provider_settings("groq", server.base_url, "groq")])
            answer = await llm.ask(messages=[Message.user_message("hi")], stream=True)

        assert answer == "streamed answer"
        assert provider.requests[0]["stream"] is True


class TestFailover:
    @pytest.mark.asyncio
    async def test_rate_limit_falls_back_to_the_next_provider(self):
        groq = MockProvider(
            "groq",
            "groq",
            reply="groq answer",
            failures=[{"status": 429, "body": {"error": {"message": "rate limit"}}}],
        )
        gemini = MockProvider("gemini", "gemini", reply="gemini answer")
        with MockProviderServer(groq) as groq_server, MockProviderServer(
            gemini
        ) as gemini_server:
            llm = build_llm(
                [
                    provider_settings("groq", groq_server.base_url, "groq"),
                    provider_settings("gemini", gemini_server.base_url, "gemini"),
                ]
            )
            answer = await llm.ask(messages=[Message.user_message("hi")], stream=False)

        assert answer == "gemini answer"
        assert len(groq.requests) == 1
        assert len(gemini.requests) == 1

    @pytest.mark.asyncio
    async def test_invalid_key_falls_back_to_the_next_provider(self):
        groq = MockProvider(
            "groq",
            "groq",
            reply="groq answer",
            failures=[{"status": 401, "body": {"error": {"message": "bad key"}}}],
        )
        openrouter = MockProvider("openrouter", "openrouter", reply="openrouter answer")
        with MockProviderServer(groq) as groq_server, MockProviderServer(
            openrouter
        ) as openrouter_server:
            llm = build_llm(
                [
                    provider_settings("groq", groq_server.base_url, "groq"),
                    provider_settings(
                        "openrouter", openrouter_server.base_url, "openrouter"
                    ),
                ]
            )
            answer = await llm.ask(messages=[Message.user_message("hi")], stream=False)

        assert answer == "openrouter answer"

    @pytest.mark.asyncio
    async def test_rejected_request_falls_back_to_the_next_model(self):
        """A 400 on the preferred model must rotate to the next model."""
        groq = MockProvider(
            "groq",
            "groq",
            reply="second model answer",
            failures=[{"status": 400, "body": {"error": {"message": "bad request"}}}],
        )
        with MockProviderServer(groq) as server:
            llm = build_llm(
                [provider_settings("groq", server.base_url, "groq")],
            )
            answer = await llm.ask(messages=[Message.user_message("hi")], stream=False)

        assert answer == "second model answer"
        # Two different models were tried.
        models = [request["model"] for request in groq.requests]
        assert len(models) == 2
        assert models[0] != models[1]

    @pytest.mark.asyncio
    async def test_truncated_answer_falls_back_to_the_next_model(self):
        groq = MockProvider(
            "groq",
            "groq",
            reply="",
            empty_content=True,
            finish_reason="length",
        )
        with MockProviderServer(groq) as server:
            llm = build_llm(
                [provider_settings("groq", server.base_url, "groq")],
            )
            # Every model of the mock is truncated -> the error surfaces.
            with pytest.raises(Exception):
                await llm.ask(messages=[Message.user_message("hi")], stream=False)

        # The client tried more than one model before giving up.
        assert len(groq.requests) > 1

    @pytest.mark.asyncio
    async def test_all_providers_failing_raises_the_last_error(self):
        groq = MockProvider(
            "groq",
            "groq",
            always_fail={"status": 429, "body": {"error": {"message": "rate limit"}}},
        )
        with MockProviderServer(groq) as server:
            llm = build_llm(
                [provider_settings("groq", server.base_url, "groq")],
            )
            with pytest.raises(Exception):
                await llm.ask(messages=[Message.user_message("hi")], stream=False)

    @pytest.mark.asyncio
    async def test_tool_call_falls_back_to_the_next_provider(self):
        groq = MockProvider(
            "groq",
            "groq",
            failures=[{"status": 429, "body": {"error": {"message": "rate limit"}}}],
        )
        gemini = MockProvider("gemini", "gemini", reply="gemini tool answer")
        with MockProviderServer(groq) as groq_server, MockProviderServer(
            gemini
        ) as gemini_server:
            llm = build_llm(
                [
                    provider_settings("groq", groq_server.base_url, "groq"),
                    provider_settings("gemini", gemini_server.base_url, "gemini"),
                ]
            )
            response = await llm.ask_tool(
                messages=[Message.user_message("hi")],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "python_execute",
                            "description": "run python",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                            },
                        },
                    }
                ],
            )

        assert response is not None
        assert response.content == "gemini tool answer"


class TestProviderChain:
    @pytest.mark.asyncio
    async def test_a_single_working_provider_is_enough(self):
        """Groq down, Gemini up -> the agent still works."""
        groq = MockProvider(
            "groq",
            "groq",
            always_fail={"status": 500, "body": {"error": {"message": "boom"}}},
        )
        gemini = MockProvider("gemini", "gemini", reply="gemini works")
        with MockProviderServer(groq) as groq_server, MockProviderServer(
            gemini
        ) as gemini_server:
            llm = build_llm(
                [
                    provider_settings("groq", groq_server.base_url, "groq"),
                    provider_settings("gemini", gemini_server.base_url, "gemini"),
                ]
            )
            answer = await llm.ask(messages=[Message.user_message("hi")], stream=False)

        assert answer == "gemini works"

    @pytest.mark.asyncio
    async def test_a_provider_without_a_key_is_skipped(self):
        gemini = MockProvider("gemini", "gemini", reply="gemini works")
        with MockProviderServer(gemini) as gemini_server:
            settings = LLMSettings(
                model="auto",
                base_url="",
                api_key="",
                max_tokens=1024,
                temperature=1.0,
                api_type="",
                api_version="",
                providers=[
                    ProviderSettings(
                        name="groq",
                        base_url="http://127.0.0.1:9/v1",
                        api_key="",
                        family="groq",
                    ),
                    provider_settings("gemini", gemini_server.base_url, "gemini"),
                ],
            )
            llm = LLM("default", llm_config={"default": settings})
            answer = await llm.ask(messages=[Message.user_message("hi")], stream=False)

        assert answer == "gemini works"

    @pytest.mark.asyncio
    async def test_no_usable_provider_raises_a_clear_error(self):
        llm = build_llm(
            [
                ProviderSettings(
                    name="groq",
                    base_url="http://127.0.0.1:9/v1",
                    api_key="k",
                    family="groq",
                )
            ]
        )
        with pytest.raises(RuntimeError, match="No usable LLM provider"):
            await llm.ask(messages=[Message.user_message("hi")], stream=False)


class TestLegacyMode:
    @pytest.mark.asyncio
    async def test_a_fixed_model_is_used_without_discovery(self):
        """A fixed model must not trigger a /models request."""
        provider = MockProvider("groq", "groq", reply="legacy answer")
        with MockProviderServer(provider) as server:
            settings = LLMSettings(
                model="llama-3.3-70b-versatile",
                base_url=server.base_url,
                api_key="test-key",
                max_tokens=1024,
                temperature=1.0,
                api_type="",
                api_version="",
                providers=[
                    ProviderSettings(
                        name="default",
                        base_url=server.base_url,
                        api_key="test-key",
                        family="groq",
                        model="llama-3.3-70b-versatile",
                    )
                ],
            )
            llm = LLM("default", llm_config={"default": settings})
            answer = await llm.ask(messages=[Message.user_message("hi")], stream=False)

        assert answer == "legacy answer"
        assert provider.requests[0]["model"] == "llama-3.3-70b-versatile"

    @pytest.mark.asyncio
    async def test_the_legacy_mode_still_strips_the_message_name(self):
        provider = MockProvider("groq", "groq", reply="ok")
        with MockProviderServer(provider) as server:
            settings = LLMSettings(
                model="llama-3.3-70b-versatile",
                base_url=server.base_url,
                api_key="test-key",
                max_tokens=1024,
                temperature=1.0,
                api_type="",
                api_version="",
                providers=[
                    ProviderSettings(
                        name="default",
                        base_url=server.base_url,
                        api_key="test-key",
                        family="groq",
                        model="llama-3.3-70b-versatile",
                    )
                ],
            )
            llm = LLM("default", llm_config={"default": settings})
            await llm.ask_tool(
                messages=[
                    Message.user_message("hi"),
                    Message.tool_message(
                        content="result", name="python_execute", tool_call_id="call_1"
                    ),
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "python_execute",
                            "description": "run python",
                            "parameters": {
                                "type": "object",
                                "properties": {"code": {"type": "string"}},
                            },
                        },
                    }
                ],
            )

        for message in provider.requests[0]["messages"]:
            assert "name" not in message


class TestRetrySemantics:
    @pytest.mark.asyncio
    async def test_token_limit_is_not_retried(self):
        """A token limit is a permanent error - it must not be retried."""
        provider = MockProvider("groq", "groq", reply="ok")
        with MockProviderServer(provider) as server:
            llm = build_llm(
                [provider_settings("groq", server.base_url, "groq")],
                max_input_tokens=1,
            )
            with pytest.raises(TokenLimitExceeded):
                await llm.ask(
                    messages=[Message.user_message("a fairly long question")],
                    stream=False,
                )

        # Not a single request was sent to the provider.
        assert provider.requests == []

    @pytest.mark.asyncio
    async def test_token_limit_is_raised_directly(self):
        provider = MockProvider("groq", "groq", reply="ok")
        with MockProviderServer(provider) as server:
            llm = build_llm(
                [provider_settings("groq", server.base_url, "groq")],
                max_input_tokens=1,
            )
            with pytest.raises(TokenLimitExceeded) as excinfo:
                await llm.ask(
                    messages=[Message.user_message("a fairly long question")],
                    stream=False,
                )
        # The original error, not a RetryError wrapper.
        assert not isinstance(excinfo.value, asyncio.TimeoutError)
        assert "token limit" in str(excinfo.value).lower()
