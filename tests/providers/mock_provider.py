"""A strict, configurable OpenAI-compatible mock server.

It reproduces the request restrictions that the real providers document, so
that the agent can be exercised against realistic failures:

Groq (https://console.groq.com/docs/openai)
    * ``messages[].name``            -> 400
    * ``logprobs`` / ``logit_bias`` / ``top_logprobs`` -> 400
    * ``n != 1``                     -> 400
    * ``max_tokens`` above the model's ``max_completion_tokens`` -> 400

Gemini (https://ai.google.dev/gemini-api/docs/openai)
    * unknown model                  -> 404

OpenRouter (https://openrouter.ai/docs/api_reference/models)
    * model list carries pricing / supported_parameters / endpoints
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional


class MockProvider:
    """Behaviour configuration of one mock provider."""

    def __init__(
        self,
        name: str,
        family: str,
        models: Optional[List[Dict]] = None,
        # A list of (status_code, body) that is consumed one entry per chat
        # request. ``None`` means "always succeed".
        failures: Optional[List[Optional[Dict]]] = None,
        strict: bool = True,
        reply: Optional[str] = None,
        finish_reason: str = "stop",
        empty_content: bool = False,
        always_fail: Optional[Dict] = None,
        replies: Optional[List[Dict]] = None,
    ):
        self.name = name
        self.family = family
        self.models = models or []
        self.failures = failures or []
        self.strict = strict
        self.reply = reply
        self.finish_reason = finish_reason
        self.empty_content = empty_content
        self.always_fail = always_fail
        self.replies = list(replies or [])
        self.reply_index = 0
        self.requests: List[Dict] = []
        self.model_detail: Dict[str, Dict] = {}

    def next_failure(self) -> Optional[Dict]:
        if not self.failures:
            return None
        failure = self.failures.pop(0)
        return failure


def _groq_catalog() -> List[Dict]:
    """Mirrors the live shape of GET https://api.groq.com/openai/v1/models.

    The live response carries pricing, context window, max completion tokens,
    modalities and the supported features per model.
    """
    return [
        {
            "id": "openai/gpt-oss-120b",
            "object": "model",
            "created": 1693721698,
            "owned_by": "OpenAI",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 65536,
            "public_apps": None,
            "pricing": {"prompt": "0.00000015", "completion": "0.0000006"},
            "modalities": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_features": ["tools", "reasoning", "json_mode"],
        },
        {
            "id": "openai/gpt-oss-20b",
            "object": "model",
            "created": 1693721698,
            "owned_by": "OpenAI",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 65536,
            "public_apps": None,
            "pricing": {"prompt": "0.000000075", "completion": "0.0000003"},
            "modalities": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_features": ["tools", "reasoning", "json_mode"],
        },
        {
            "id": "qwen/qwen3-32b",
            "object": "model",
            "created": 1693721698,
            "owned_by": "Alibaba",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 40960,
            "public_apps": None,
            "pricing": {"prompt": "0", "completion": "0"},
            "modalities": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_features": ["tools", "json_mode"],
        },
        {
            "id": "meta-llama/llama-4-scout-17b-16e-instruct",
            "object": "model",
            "created": 1693721698,
            "owned_by": "Meta",
            "active": True,
            "context_window": 131072,
            "max_completion_tokens": 8192,
            "public_apps": None,
            "pricing": {"prompt": "0", "completion": "0"},
            "modalities": {
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "supported_features": ["tools", "json_mode"],
        },
        {
            "id": "whisper-large-v3-turbo",
            "object": "model",
            "created": 1728413088,
            "owned_by": "OpenAI",
            "active": True,
            "context_window": 448,
            "max_completion_tokens": 448,
            "public_apps": None,
            "pricing": {"prompt": "0.00000004", "completion": "0"},
            "modalities": {
                "input_modalities": ["audio"],
                "output_modalities": ["text"],
            },
            "supported_features": [],
        },
        {
            "id": "meta-llama/llama-guard-3-8b",
            "object": "model",
            "created": 1693721698,
            "owned_by": "Meta",
            "active": True,
            "context_window": 8192,
            "max_completion_tokens": 8192,
            "public_apps": None,
            "pricing": {"prompt": "0.00000003", "completion": "0.00000003"},
            "modalities": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_features": [],
        },
        {
            "id": "retired-model-v1",
            "object": "model",
            "created": 1693721698,
            "owned_by": "Meta",
            "active": False,
            "context_window": 8192,
            "max_completion_tokens": 8192,
            "public_apps": None,
            "pricing": {"prompt": "0", "completion": "0"},
            "modalities": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_features": ["tools"],
        },
    ]


def _gemini_catalog() -> List[Dict]:
    """Mirrors GET https://generativelanguage.googleapis.com/v1beta/openai/models."""
    return [
        {
            "id": "models/gemini-2.5-flash",
            "object": "model",
            "owned_by": "google",
            "max_completion_tokens": 65536,
        },
        {
            "id": "models/gemini-2.5-pro",
            "object": "model",
            "owned_by": "google",
            "max_completion_tokens": 65536,
        },
        {
            "id": "models/gemini-embedding-2",
            "object": "model",
            "owned_by": "google",
        },
        {
            "id": "models/gemini-2.5-flash-image",
            "object": "model",
            "owned_by": "google",
        },
    ]


def _openrouter_catalog() -> List[Dict]:
    """Mirrors GET https://openrouter.ai/api/v1/models (live shape)."""
    return [
        {
            "id": "meta-llama/llama-3.3-70b-instruct:free",
            "name": "Meta: Llama 3.3 70B Instruct (free)",
            "context_length": 131072,
            "architecture": {
                "modality": "text->text",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "pricing": {"prompt": "0", "completion": "0"},
            "top_provider": {"context_length": 131072, "max_completion_tokens": 32768},
            "supported_parameters": [
                "max_tokens",
                "temperature",
                "tool_choice",
                "tools",
            ],
            "endpoints": ["/api/v1/chat/completions"],
        },
        {
            "id": "google/gemini-2.5-flash:free",
            "name": "Google: Gemini 2.5 Flash (free)",
            "context_length": 1048576,
            "architecture": {
                "modality": "text+image->text",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "pricing": {"prompt": "0", "completion": "0"},
            "top_provider": {"context_length": 1048576, "max_completion_tokens": 65536},
            "supported_parameters": [
                "max_tokens",
                "temperature",
                "tool_choice",
                "tools",
            ],
            "endpoints": ["/api/v1/chat/completions"],
        },
        {
            "id": "anthropic/claude-sonnet-4.5",
            "name": "Anthropic: Claude Sonnet 4.5",
            "context_length": 200000,
            "architecture": {
                "modality": "text->text",
                "input_modalities": ["text"],
                "output_modalities": ["text"],
            },
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            "top_provider": {"context_length": 200000, "max_completion_tokens": 64000},
            "supported_parameters": [
                "max_tokens",
                "temperature",
                "tool_choice",
                "tools",
            ],
            "endpoints": ["/api/v1/chat/completions"],
        },
        {
            "id": "openai/gpt-4o-mini:batch",
            "name": "OpenAI: GPT-4o mini (batch)",
            "context_length": 128000,
            "pricing": {"prompt": "0", "completion": "0"},
            "supported_parameters": ["max_tokens"],
            "endpoints": ["/api/v1/chat/completions"],
        },
        {
            "id": "openrouter/auto",
            "name": "Auto",
            "context_length": 1000000,
            "pricing": {"prompt": "-1", "completion": "-1"},
            "supported_parameters": ["max_tokens", "tools"],
            "endpoints": ["/api/v1/chat/completions"],
        },
    ]


DEFAULT_CATALOGS = {
    "groq": _groq_catalog,
    "gemini": _gemini_catalog,
    "openrouter": _openrouter_catalog,
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        """Keep the pytest output clean."""

    @property
    def provider(self) -> MockProvider:
        return self.server.provider  # type: ignore[attr-defined]

    def _send_json(self, status: int, payload: Dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        provider = self.provider
        catalog = (
            provider.models if provider.models else DEFAULT_CATALOGS[provider.family]()
        )
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(200, {"object": "list", "data": catalog})
            return

        # GET /models/{id}
        if "/models/" in self.path:
            model_id = self.path.rsplit("/", 1)[-1]
            for entry in catalog:
                if entry["id"] == model_id or entry["id"].endswith(f"/{model_id}"):
                    detail = dict(entry)
                    detail.setdefault("max_completion_tokens", 8192)
                    self._send_json(200, detail)
                    return
            self._send_json(
                404,
                {
                    "error": {
                        "message": f"model '{model_id}' not found",
                        "type": "not_found",
                    }
                },
            )
            return

        self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        provider = self.provider
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError:
            self._send_json(400, {"error": {"message": "invalid JSON"}})
            return

        provider.requests.append(payload)

        if provider.strict:
            error = self._validate(payload)
            if error is not None:
                self._send_json(
                    error["status"],
                    {
                        "error": {
                            "message": error["message"],
                            "type": "invalid_request_error",
                        }
                    },
                )
                return

        failure = provider.always_fail or provider.next_failure()
        if failure is not None:
            self._send_json(failure.get("status", 500), failure.get("body", {}))
            return

        if payload.get("stream"):
            self._send_stream(self._completion(payload))
            return

        self._send_json(200, self._completion(payload))

    def _send_stream(self, body: Dict) -> None:
        """Send a minimal server-sent-event stream."""
        message = body["choices"][0]["message"]
        chunks = []
        content = message.get("content") or ""
        if content:
            chunks.append(
                {
                    "id": body["id"],
                    "object": "chat.completion.chunk",
                    "created": body["created"],
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content},
                            "finish_reason": None,
                        }
                    ],
                }
            )
        # Trailing chunk without choices (usage only) - providers do this.
        chunks.append(
            {
                "id": body["id"],
                "object": "chat.completion.chunk",
                "created": body["created"],
                "model": body["model"],
                "choices": [],
                "usage": body.get("usage"),
            }
        )

        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        payload += "data: [DONE]\n\n"
        raw = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _validate(self, payload: Dict) -> Optional[Dict]:
        provider = self.provider

        if provider.family == "groq":
            for message in payload.get("messages") or []:
                if "name" in message:
                    return {
                        "status": 400,
                        "message": (
                            "Unsupported parameter: 'messages[].name' is not "
                            "supported with this model."
                        ),
                    }
            for unsupported in ("logprobs", "logit_bias", "top_logprobs"):
                if payload.get(unsupported) is not None:
                    return {
                        "status": 400,
                        "message": f"Unsupported parameter: '{unsupported}'.",
                    }
            if payload.get("n") not in (None, 1):
                return {"status": 400, "message": "n must be 1."}
            limit = self._max_completion_tokens(payload.get("model"))
            requested = payload.get("max_tokens") or payload.get(
                "max_completion_tokens"
            )
            if limit and requested and requested > limit:
                return {
                    "status": 400,
                    "message": (
                        f"Requested max_tokens ({requested}) is greater than the "
                        f"model's maximum completion tokens ({limit})."
                    ),
                }

        if provider.family == "gemini":
            model = payload.get("model") or ""
            if not model.startswith("gemini-"):
                return {
                    "status": 404,
                    "message": f"models/{model} is not found for API version v1beta.",
                }

        return None

    def _max_completion_tokens(self, model_id: Optional[str]) -> Optional[int]:
        if not model_id:
            return None
        provider = self.provider
        for entry in provider.models or DEFAULT_CATALOGS[provider.family]():
            if entry["id"] == model_id or entry["id"].endswith(f"/{model_id}"):
                return entry.get("max_completion_tokens", 8192)
        return 8192

    def _completion(self, payload: Dict) -> Dict:
        model = payload.get("model", "mock")
        tools = payload.get("tools") or []
        content = self.provider.reply if self.provider.reply is not None else "ok"
        if self.provider.empty_content:
            content = ""
        message = {"role": "assistant", "content": content}
        finish_reason = self.provider.finish_reason
        tool_calls = None

        # A scripted reply overrides the defaults. It allows a test to drive
        # a full agent run (tool call -> tool result -> final answer).
        if self.provider.replies:
            index = min(self.provider.reply_index, len(self.provider.replies) - 1)
            script = self.provider.replies[index]
            self.provider.reply_index += 1
            if "content" in script:
                content = script["content"]
                message["content"] = content
            if "tool_calls" in script:
                tool_calls = script["tool_calls"]
            if "finish_reason" in script:
                finish_reason = script["finish_reason"]

        if tool_calls is None and tools and not self.provider.replies:
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": tools[0]["function"]["name"],
                        "arguments": json.dumps({"code": "print(1)"}),
                    },
                }
            ]
        if tool_calls:
            message["tool_calls"] = tool_calls
        body = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1700000000,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": ("tool_calls" if tool_calls else finish_reason),
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }
        if self.provider.family == "openrouter":
            body["openrouter_metadata"] = {"requested": model, "attempt": 1}
        return body


class MockProviderServer:
    """Context manager that runs a :class:`MockProvider` on a random port."""

    def __init__(self, provider: MockProvider):
        self.provider = provider
        # A threading server is required: the OpenAI client keeps the
        # connection alive (HTTP/1.1) and may open several connections.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.provider = provider  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> MockProviderServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._server.shutdown()
        self._server.server_close()
