"""Provider registry, live model discovery and automatic model selection.

Every provider used by OpenManus is reachable through the OpenAI-compatible
Chat Completions API, so a provider is fully described by:

* a ``base_url``
* an ``api_key``
* a list of candidate models

This module discovers the models that are actually available *right now* from
the provider's own ``GET {base_url}/models`` endpoint, filters out the ones
that cannot be used for a tool-calling agent, ranks them and hands the best
candidates back to :mod:`app.llm`, which then rotates to the next candidate
whenever a provider fails (rate limit, outage, model retired, bad request).

Endpoint contracts (verified against the providers' own documentation):

Groq -- https://console.groq.com/docs/api-reference
    ``GET https://api.groq.com/openai/v1/models`` ->
    ``{"object": "list", "data": [{"id", "object", "created", "owned_by",
    "active", "context_window", "public_apps"}]}``
    ``GET .../models/{id}`` additionally returns ``max_completion_tokens``.
    Unsupported request fields (400 error): ``logprobs``, ``logit_bias``,
    ``top_logprobs``, ``messages[].name`` and ``n != 1``.

Gemini -- https://ai.google.dev/gemini-api/docs/openai
    ``GET https://generativelanguage.googleapis.com/v1beta/openai/models`` ->
    OpenAI style ``{"object": "list", "data": [{"id", "object", "owned_by"}]}``
    (ids may be prefixed with ``models/``). Authentication is
    ``Authorization: Bearer <key>``.

OpenRouter -- https://openrouter.ai/docs/api_reference/models
    ``GET https://openrouter.ai/api/v1/models`` ->
    ``{"data": [{"id", "context_length", "architecture",
    "pricing": {"prompt", "completion"}, "top_provider",
    "supported_parameters", "endpoints"}]}``
    A model is free when both ``pricing.prompt`` and ``pricing.completion``
    are ``"0"``. Dynamic routers use the sentinel price ``"-1"``.
"""

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

from app.logger import logger


DEFAULT_CACHE_TTL_SECONDS = 6 * 3600
DISCOVERY_TIMEOUT_SECONDS = 20.0


def _cache_dir() -> Path:
    override = os.environ.get("OPENMANUS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "openmanus"


def _cache_ttl() -> int:
    try:
        return int(
            os.environ.get("OPENMANUS_MODEL_CACHE_TTL", DEFAULT_CACHE_TTL_SECONDS)
        )
    except ValueError:
        return DEFAULT_CACHE_TTL_SECONDS


@dataclass
class ProviderDefaults:
    """Static description of a well known OpenAI-compatible provider."""

    name: str
    base_url: str
    api_key_env: List[str]
    family: str
    default_headers: Dict[str, str] = field(default_factory=dict)


BUILTIN_PROVIDERS: Dict[str, ProviderDefaults] = {
    "groq": ProviderDefaults(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env=["GROQ_API_KEY"],
        family="groq",
    ),
    "gemini": ProviderDefaults(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        family="gemini",
    ),
    "openrouter": ProviderDefaults(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env=["OPENROUTER_API_KEY"],
        family="openrouter",
        default_headers={
            "HTTP-Referer": "https://github.com/FoundationAgents/OpenManus",
            "X-Title": "OpenManus",
        },
    ),
}


# Ids that never denote a chat model an agent can drive with tools.
_NON_CHAT_PATTERNS = (
    "whisper",
    "tts",
    "text-to-speech",
    "speech",
    "orpheus",
    "playai",
    "guard",
    "moderation",
    "embed",
    "clip",
    "rerank",
    "imagen",
    "veo",
    "aqa",
    "ocr-",
)

# Gemini exposes dedicated image/video *generation* models
# (gemini-2.5-flash-image, gemini-3-pro-image-preview, ...). They answer on
# the images/video endpoints, not on chat completions.
_IMAGE_GENERATION_PATTERN = re.compile(
    r"image(-generation|-preview)?$|image-generation"
)


# The family of a well known provider is determined by its *name*, not by its
# base URL: a user may point a provider at a proxy, a gateway or a mock
# server, and the catalogue parsing (free models, tool support, modalities)
# must still be applied.
FAMILY_BY_NAME: Dict[str, str] = {
    "groq": "groq",
    "gemini": "gemini",
    "google": "gemini",
    "openrouter": "openrouter",
}


def infer_family(base_url: str, name: str = "") -> str:
    """Identify the provider family from its name or base URL."""
    if name:
        family = FAMILY_BY_NAME.get(name.lower())
        if family:
            return family
    lowered = (base_url or "").lower()
    if "api.groq.com" in lowered:
        return "groq"
    if "generativelanguage.googleapis.com" in lowered:
        return "gemini"
    if "openrouter.ai" in lowered:
        return "openrouter"
    return "generic"


@dataclass
class ModelInfo:
    """A single model discovered from a provider."""

    id: str
    context_window: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    supports_tools: Optional[bool] = None
    supports_images: bool = False
    is_free: bool = False
    cost: Optional[float] = None
    quality: Optional[float] = None
    reasoning: bool = False
    raw: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "context_window": self.context_window,
            "max_completion_tokens": self.max_completion_tokens,
            "supports_tools": self.supports_tools,
            "supports_images": self.supports_images,
            "is_free": self.is_free,
            "cost": self.cost,
            "quality": self.quality,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelInfo":
        return cls(
            id=data["id"],
            context_window=data.get("context_window"),
            max_completion_tokens=data.get("max_completion_tokens"),
            supports_tools=data.get("supports_tools"),
            supports_images=bool(data.get("supports_images")),
            is_free=bool(data.get("is_free")),
            cost=data.get("cost"),
            quality=data.get("quality"),
            reasoning=bool(data.get("reasoning")),
        )


@dataclass
class ProviderSpec:
    """Everything needed to talk to one provider."""

    name: str
    base_url: str
    api_key: str
    model: str = "auto"
    prefer: List[str] = field(default_factory=list)
    free_only: bool = True
    max_candidates: int = 3
    extra_headers: Dict[str, str] = field(default_factory=dict)
    # Empty means "derive it from the name / base URL".
    family: str = ""

    def __post_init__(self) -> None:
        # The settings may come from a pydantic model where the optional
        # fields are None.
        if self.extra_headers is None:
            self.extra_headers = {}
        if self.prefer is None:
            self.prefer = []
        if not self.family:
            self.family = infer_family(self.base_url, self.name)

    @property
    def models_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/models"

    @property
    def headers(self) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        headers.update(self.extra_headers)
        return headers


@dataclass
class Candidate:
    """A provider plus a concrete model to try."""

    provider: ProviderSpec
    model: str
    max_completion_tokens: Optional[int] = None
    supports_images: bool = False
    reasoning: bool = False

    @property
    def label(self) -> str:
        return f"{self.provider.name}/{self.model}"


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def _looks_like_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(pattern in lowered for pattern in _NON_CHAT_PATTERNS)


def _parse_price(value) -> Optional[float]:
    """Parse a per-token price. ``None`` means "not advertised"."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_groq(payload: Dict) -> List[ModelInfo]:
    """Parse Groq's model list.

    The live ``GET /openai/v1/models`` response carries, per model:
    ``id``, ``active``, ``context_window``, ``max_completion_tokens``,
    ``pricing`` (``prompt``/``completion``/``input_cache_read``),
    ``modalities`` (``input_modalities``/``output_modalities``) and
    ``supported_features`` (e.g. ``["tools", "reasoning", "json_mode"]``).

    Older responses only contain ``id``/``context_window``; those are handled
    gracefully (free is assumed, tool support stays unknown).
    """
    models: List[ModelInfo] = []
    for entry in payload.get("data") or []:
        model_id = entry.get("id")
        if not model_id or not _looks_like_chat_model(model_id):
            continue
        if entry.get("active") is False:
            continue

        # --- pricing ---------------------------------------------------- #
        pricing = entry.get("pricing") or {}
        prompt_price = _parse_price(pricing.get("prompt"))
        completion_price = _parse_price(pricing.get("completion"))
        if prompt_price is None and completion_price is None:
            # No pricing published -> the model is usable with a free key.
            is_free = True
        else:
            is_free = (prompt_price or 0.0) == 0.0 and (completion_price or 0.0) == 0.0

        # --- capabilities ------------------------------------------------ #
        features = entry.get("supported_features")
        supports_tools = None
        if isinstance(features, list):
            supports_tools = any(
                feature in features
                for feature in ("tools", "tool_use", "function_calling")
            )

        modalities = entry.get("modalities") or entry.get("architecture") or {}
        input_modalities = modalities.get("input_modalities") or []
        supports_images = "image" in input_modalities

        models.append(
            ModelInfo(
                id=model_id,
                context_window=entry.get("context_window"),
                max_completion_tokens=entry.get("max_completion_tokens"),
                supports_tools=supports_tools,
                supports_images=supports_images,
                is_free=is_free,
                cost=(
                    (prompt_price or 0.0) + (completion_price or 0.0)
                    if prompt_price is not None or completion_price is not None
                    else None
                ),
                reasoning=isinstance(features, list) and "reasoning" in features,
                raw=entry,
            )
        )
    return models


def _parse_gemini(payload: Dict) -> List[ModelInfo]:
    """Parse Gemini's OpenAI-compatible model list.

    ``GET https://generativelanguage.googleapis.com/v1beta/openai/models``
    returns OpenAI style entries; the ids may carry a ``models/`` prefix which
    must be removed before they are used in a chat completion request.
    """
    models: List[ModelInfo] = []
    for entry in payload.get("data") or []:
        model_id = entry.get("id") or ""
        # Gemini may return ids prefixed with "models/".
        if model_id.startswith("models/"):
            model_id = model_id[len("models/") :]
        if not model_id or not _looks_like_chat_model(model_id):
            continue
        # Image *generation* models cannot be used as chat models.
        if _IMAGE_GENERATION_PATTERN.search(model_id.lower()):
            continue

        if "input_modalities" in entry:
            modalities = entry.get("input_modalities") or []
            supports_images = "image" in modalities
        else:
            supports_images = model_id.lower().startswith("gemini-")

        models.append(
            ModelInfo(
                id=model_id,
                context_window=entry.get("context_window"),
                max_completion_tokens=entry.get("max_completion_tokens"),
                supports_tools=None,
                supports_images=supports_images,
                is_free=True,
                reasoning=model_id.lower().startswith("gemini-2.5")
                or "thinking" in model_id.lower(),
                raw=entry,
            )
        )
    return models


def _parse_openrouter(payload: Dict) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    for entry in payload.get("data") or []:
        model_id = entry.get("id")
        if not model_id or not _looks_like_chat_model(model_id):
            continue
        # Batch variants are asynchronous and live under a different endpoint.
        if model_id.endswith(":batch"):
            continue
        # Rolling aliases ("~vendor/model") are not stable ids.
        if model_id.startswith("~"):
            continue

        pricing = entry.get("pricing") or {}
        try:
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))
        except (TypeError, ValueError):
            prompt_price = completion_price = 0.0

        # "-1" is the sentinel price of OpenRouter's dynamic routers
        # (openrouter/auto, ...). They are routers, not concrete models.
        if prompt_price < 0 or completion_price < 0:
            continue

        is_free = prompt_price == 0 and completion_price == 0

        supported = entry.get("supported_parameters")
        supports_tools = None
        if isinstance(supported, list):
            supports_tools = "tools" in supported

        endpoints = entry.get("endpoints")
        if isinstance(endpoints, list) and endpoints:
            if "/api/v1/chat/completions" not in endpoints:
                continue

        architecture = entry.get("architecture") or {}
        input_modalities = architecture.get("input_modalities") or []
        supports_images = "image" in input_modalities

        top_provider = entry.get("top_provider") or {}
        quality = _openrouter_quality(entry)

        models.append(
            ModelInfo(
                id=model_id,
                context_window=entry.get("context_length"),
                max_completion_tokens=top_provider.get("max_completion_tokens"),
                supports_tools=supports_tools,
                supports_images=supports_images,
                is_free=is_free,
                cost=prompt_price + completion_price,
                quality=quality,
                raw=entry,
            )
        )
    return models


def _openrouter_quality(entry: Dict) -> Optional[float]:
    """Extract a comparable quality signal from OpenRouter's benchmark data."""
    benchmarks = entry.get("benchmarks") or {}
    artificial = benchmarks.get("artificial_analysis") or {}
    for key in ("intelligence_index", "coding_index", "agentic_index"):
        value = artificial.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _parse_generic(payload: Dict) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    for entry in payload.get("data") or []:
        model_id = entry.get("id")
        if not model_id or not _looks_like_chat_model(model_id):
            continue
        models.append(
            ModelInfo(
                id=model_id,
                context_window=entry.get("context_window"),
                max_completion_tokens=entry.get("max_completion_tokens"),
                supports_tools=None,
                is_free=True,
                raw=entry,
            )
        )
    return models


_PARSERS = {
    "groq": _parse_groq,
    "gemini": _parse_gemini,
    "openrouter": _parse_openrouter,
    "generic": _parse_generic,
}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def _cache_file(provider_name: str) -> Path:
    return _cache_dir() / "models" / f"{provider_name}.json"


def read_cached_models(provider_name: str) -> Optional[List[ModelInfo]]:
    path = _cache_file(provider_name)
    try:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > _cache_ttl():
            return None
        payload = json.loads(path.read_text())
        return [ModelInfo.from_dict(item) for item in payload.get("models", [])]
    except (OSError, ValueError, KeyError):
        return None


def write_cached_models(provider_name: str, models: Sequence[ModelInfo]) -> None:
    path = _cache_file(provider_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "fetched_at": time.time(),
                    "models": [model.to_dict() for model in models],
                },
                indent=1,
            )
        )
    except OSError:
        logger.debug(f"Could not cache the model list for provider '{provider_name}'")


def fetch_models(spec: ProviderSpec) -> List[ModelInfo]:
    """Fetch the live model catalog of a provider."""
    parser = _PARSERS.get(spec.family, _parse_generic)
    try:
        response = requests.get(
            spec.models_url,
            headers=spec.headers,
            timeout=DISCOVERY_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the model list of provider '{spec.name}': {exc}"
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"Provider '{spec.name}' answered {response.status_code} on "
            f"{spec.models_url}: {response.text[:300]}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Provider '{spec.name}' returned a non-JSON model list: {exc}"
        ) from exc

    models = parser(payload)
    if not models:
        raise RuntimeError(
            f"Provider '{spec.name}' did not advertise any usable chat model."
        )
    return models


def discover_models(spec: ProviderSpec, use_cache: bool = True) -> List[ModelInfo]:
    """Return the provider's usable chat models, using the on-disk cache."""
    if use_cache:
        cached = read_cached_models(spec.name)
        if cached:
            logger.info(
                f"Using cached model list for provider '{spec.name}' "
                f"({len(cached)} models)"
            )
            return cached

    models = fetch_models(spec)
    write_cached_models(spec.name, models)
    logger.info(f"Discovered {len(models)} models on provider '{spec.name}'")
    return models


def fetch_model_detail(spec: ProviderSpec, model_id: str) -> Dict:
    """Fetch the details of a single model (``GET {base_url}/models/{id}``).

    Groq publishes ``max_completion_tokens`` for a model here. Knowing the
    limit lets OpenManus clamp ``max_tokens`` instead of asking for more
    output tokens than the model can ever produce (which is a 400 error).
    """
    url = f"{spec.base_url.rstrip('/')}/models/{model_id}"
    try:
        response = requests.get(
            url, headers=spec.headers, timeout=DISCOVERY_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:
        logger.debug(f"Could not read model details for '{model_id}': {exc}")
        return {}
    if response.status_code != 200:
        return {}
    try:
        return response.json()
    except ValueError:
        return {}


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #


def _parameter_scale(model_id: str) -> float:
    """Rough 'how big is the model' signal derived from the id (e.g. 70b)."""
    match = re.search(r"(\d{2,3})b(?![a-z0-9])", model_id.lower())
    if not match:
        return 0.0
    return min(int(match.group(1)) / 100.0, 2.0)


def _heuristic_score(model: ModelInfo) -> float:
    """Generic quality heuristic used when no benchmark data is available."""
    lowered = model.id.lower()
    score = _parameter_scale(lowered)
    if "flash" in lowered:
        score += 1.0
    if re.search(r"\bpro\b|(^|-)pro(-|$)", lowered):
        score += 0.6
    if "instruct" in lowered or "chat" in lowered:
        score += 0.8
    if "versatile" in lowered or "instant" in lowered:
        score += 0.4
    if any(marker in lowered for marker in ("mini", "nano", "lite", "tiny")):
        score -= 0.8
    if any(marker in lowered for marker in ("preview", "experimental", "-exp")):
        score -= 0.2
    return score


def rank_models(
    models: Sequence[ModelInfo],
    prefer: Sequence[str] = (),
    free_only: bool = True,
) -> List[ModelInfo]:
    """Order models from most to least suitable for a tool-calling agent.

    The order is:

    1. models explicitly requested through ``prefer`` (in the given order)
    2. models that are free (and, among those, the cheapest ones first)
    3. the remaining models, best heuristic score first

    ``free_only`` restricts the result to zero-cost models whenever the
    provider advertises at least one of them.
    """
    candidates = list(models)

    if free_only:
        free = [model for model in candidates if model.is_free]
        if free:
            candidates = free
        else:
            logger.warning(
                "No free model advertised by the provider - falling back to the "
                "full catalogue."
            )

    # Models that cannot do tool calling are useless for an agent.
    tool_capable = [model for model in candidates if model.supports_tools is not False]
    if tool_capable:
        candidates = tool_capable

    # 1. explicit user preferences win, in the order given
    preferred: List[ModelInfo] = []
    remaining = list(candidates)
    for wanted in prefer:
        for model in list(remaining):
            if model.id == wanted or model.id.endswith(f"/{wanted}"):
                preferred.append(model)
                remaining.remove(model)

    # 2. price first: a free/cheap model is the point of the exercise.
    # 3. benchmark score when the provider publishes one
    # 4. generic heuristic + context window as tie breakers
    #
    # All three are combined into a single key so that the relative order of
    # the criteria is well defined (successive sorts would destroy it).
    remaining.sort(key=_ranking_key)

    return preferred + remaining


def _ranking_key(model: ModelInfo):
    """Sort key: cheapest first, then best benchmark score, then heuristics."""
    quality = model.quality if model.quality is not None else 0.0
    cost = model.cost if model.cost is not None else 0.0
    return (
        cost,
        -quality,
        _heuristic_score(model),
        model.context_window or 0,
    )


def select_candidates(
    spec: ProviderSpec,
    use_cache: bool = True,
) -> List[Candidate]:
    """Build the ordered list of (provider, model) candidates to try."""
    if spec.model and spec.model != "auto":
        return [Candidate(provider=spec, model=spec.model)]

    models = discover_models(spec, use_cache=use_cache)
    ranked = rank_models(models, prefer=spec.prefer, free_only=spec.free_only)
    limit = max(1, spec.max_candidates)
    selected = ranked[:limit]
    if not selected:
        raise RuntimeError(f"Provider '{spec.name}' has no usable model.")

    # The first candidate is the one that will be used most often - make sure
    # its output limit is known so that ``max_tokens`` can be clamped.
    if selected[0].max_completion_tokens is None:
        detail = fetch_model_detail(spec, selected[0].id)
        limit_value = detail.get("max_completion_tokens")
        if isinstance(limit_value, int) and limit_value > 0:
            selected[0].max_completion_tokens = limit_value

    return [
        Candidate(
            provider=spec,
            model=model.id,
            max_completion_tokens=model.max_completion_tokens,
            supports_images=model.supports_images,
            reasoning=model.reasoning,
        )
        for model in selected
    ]
