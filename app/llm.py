import asyncio
import inspect
import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage

from app.config import LLMSettings, config
from app.exceptions import OutputTruncated, TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.providers import Candidate, ProviderSpec, select_candidates
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


REASONING_MODELS = [
    "o1",
    "o1-mini",
    "o1-preview",
    "o1-pro",
    "o3",
    "o3-mini",
    "o4-mini",
]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]

# Model-name prefixes that identify multimodal (image-capable) models. This keeps
# vision support working for third-party OpenAI-compatible providers such as
# Google Gemini, Groq (llama-4-scout) and OpenRouter, where the model id does
# not match the upstream OpenAI/Anthropic ids.
MULTIMODAL_MODEL_PREFIXES = (
    "gemini-",  # every Gemini model accepts image input
    "gpt-4o",
    "gpt-4.1",
    "gpt-5",
    "claude-3",
    "claude-4",
    "llama-4-scout",
    "llama-4-maverick",
    "qwen2.5-vl",
    "qwen3-vl",
    "internvl",
    "pixtral",
    "llava",
)


def normalize_model_id(model: str) -> str:
    """Strip provider routing prefixes (e.g. OpenRouter's ``openai/o3-mini``)."""
    if not model:
        return ""
    return model.split("/")[-1].strip().lower()


def is_reasoning_model(model: str) -> bool:
    """Return True when the model uses the ``max_completion_tokens`` parameter."""
    normalized = normalize_model_id(model)
    if not normalized:
        return False
    return any(
        normalized == candidate or normalized.startswith(candidate)
        for candidate in REASONING_MODELS
    )


def model_supports_images(model: str) -> bool:
    """Best-effort detection of image (multimodal) input support."""
    normalized = normalize_model_id(model)
    if not normalized:
        return False
    if normalized in MULTIMODAL_MODELS:
        return True
    return any(normalized.startswith(prefix) for prefix in MULTIMODAL_MODEL_PREFIXES)


class ApproximateTokenizer:
    """Offline fallback tokenizer.

    ``tiktoken`` downloads its BPE tables from ``openaipublic.blob.core.windows.net``
    on first use. When that host is unreachable (restricted egress, offline
    notebooks, air-gapped CI) the download raises and the agent cannot start.
    This tokenizer approximates the OpenAI rule of thumb of ~4 characters per
    token so that token accounting keeps working without network access.
    """

    CHARS_PER_TOKEN = 4

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        return [0] * max(
            1, (len(text) + self.CHARS_PER_TOKEN - 1) // self.CHARS_PER_TOKEN
        )

    def decode(self, tokens: List[int]) -> str:
        return ""


def load_tokenizer(model: str):
    """Build a tokenizer, degrading gracefully when tiktoken data is unavailable."""
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:  # unknown model id
        pass

    for encoding_name in ("cl100k_base", "o200k_base"):
        try:
            return tiktoken.get_encoding(encoding_name)
        except Exception:  # download failed / blocked
            continue

    logger.warning(
        "Unable to load a tiktoken encoding (offline or blocked download). "
        "Falling back to an approximate tokenizer; token counts will be rough."
    )
    return ApproximateTokenizer()


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    """OpenAI-compatible chat client with automatic provider/model failover.

    The client does not require the user to pick a model. When the
    configuration declares one or more providers (see
    :class:`app.config.ProviderSettings`), the models that are *actually*
    available right now are discovered from the providers' own ``/models``
    endpoints and turned into an ordered list of candidates. The first
    candidate is used; whenever it fails (rate limit, outage, retired model,
    rejected request, exhausted output budget) the next candidate is used.

    This makes OpenManus work with a single API key for Groq, Gemini or
    OpenRouter - and with several of them at the same time.
    """

    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            llm_config = llm_config or config.llm
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.config_name = config_name
            self.llm_config = llm_config

            self.configured_max_tokens = llm_config.max_tokens
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url
            self.extra_headers = llm_config.extra_headers
            self.send_message_name = bool(
                getattr(llm_config, "send_message_name", False)
            )
            self.max_retries = max(1, int(getattr(llm_config, "max_retries", 2)))

            # Add token counting related attributes
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            self._provider_specs = [
                ProviderSpec(**provider.model_dump())
                for provider in getattr(llm_config, "providers", []) or []
            ]
            self._candidates: List[Candidate] = []
            self._candidate_index = 0
            self._candidates_ready = False
            self._candidate_lock = asyncio.Lock()
            self._legacy_client = False
            self._clients: Dict[str, object] = {}

            self.model = llm_config.model
            self.is_reasoning_model = is_reasoning_model(self.model)

            # Image support can be forced on/off from the config file; when it is
            # not set we infer it from the model name.
            configured_image_support = getattr(llm_config, "supports_images", None)
            self.supports_images = (
                configured_image_support
                if configured_image_support is not None
                else model_supports_images(self.model)
            )

            # Initialize tokenizer (falls back to an offline approximation)
            self.tokenizer = load_tokenizer(self.model)

            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
                self._legacy_client = True
            elif self.api_type == "aws":
                # Imported lazily so boto3 is only required for Bedrock users.
                from app.bedrock import BedrockClient

                self.client = BedrockClient()
                self._legacy_client = True
            else:
                self.client = self._build_client(self.api_key, self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)

    # ------------------------------------------------------------------ #
    # Provider / model resolution
    # ------------------------------------------------------------------ #

    def _build_client(self, api_key: str, base_url: str):
        """Create (and cache) an OpenAI-compatible client for a provider.

        ``max_retries=0`` is intentional: the failover logic of this class
        decides how often a provider is retried. Letting the SDK retry on its
        own would silently multiply the number of requests against a provider
        that is already rate limited.
        """
        client_kwargs = {"api_key": api_key, "base_url": base_url, "max_retries": 0}
        headers = self.extra_headers
        if headers:
            client_kwargs["default_headers"] = headers
        return AsyncOpenAI(**client_kwargs)

    def _client_for(self, provider: ProviderSpec):
        key = f"{provider.name}|{provider.base_url}|{provider.api_key}"
        client = self._clients.get(key)
        if client is None:
            client = AsyncOpenAI(
                api_key=provider.api_key,
                base_url=provider.base_url,
                max_retries=0,
                **(
                    {"default_headers": provider.extra_headers}
                    if provider.extra_headers
                    else {}
                ),
            )
            self._clients[key] = client
        return client

    async def _ensure_candidates(self) -> None:
        """Resolve the usable (provider, model) candidates exactly once."""
        if self._candidates_ready or self._legacy_client:
            return
        async with self._candidate_lock:
            if self._candidates_ready:
                return
            await self._discover_candidates()
            self._candidates_ready = True

    async def _discover_candidates(self) -> None:
        candidates: List[Candidate] = []
        problems: List[str] = []

        for spec in self._provider_specs:
            if not spec.api_key:
                problems.append(f"{spec.name}: no API key configured")
                continue
            if not spec.base_url:
                problems.append(f"{spec.name}: no base_url configured")
                continue
            try:
                provider_candidates = await asyncio.to_thread(select_candidates, spec)
            except Exception as exc:
                problems.append(f"{spec.name}: {exc}")
                logger.warning(f"Provider '{spec.name}' is unusable: {exc}")
                continue

            candidates.extend(provider_candidates)
            logger.info(
                f"Provider '{spec.name}' offers: "
                f"{[candidate.model for candidate in provider_candidates]}"
            )

        if not candidates:
            raise RuntimeError(
                "No usable LLM provider could be reached. " + "; ".join(problems)
            )

        self._candidates = candidates
        self._activate(0)
        logger.info(
            "Active provider chain: " f"{[candidate.label for candidate in candidates]}"
        )

    def _activate(self, index: int) -> None:
        """Switch the client, the model and the token limits to a candidate."""
        candidate = self._candidates[index]
        self._candidate_index = index
        provider = candidate.provider

        self.model = candidate.model
        self.base_url = provider.base_url
        self.api_key = provider.api_key
        self.extra_headers = provider.extra_headers or None
        self.is_reasoning_model = is_reasoning_model(candidate.model)

        configured_image_support = getattr(self.llm_config, "supports_images", None)
        self.supports_images = (
            configured_image_support
            if configured_image_support is not None
            else (candidate.supports_images or model_supports_images(candidate.model))
        )

        # Never ask for more output tokens than the model allows.
        limit = candidate.max_completion_tokens
        self.max_tokens = (
            min(self.configured_max_tokens, limit)
            if limit
            else self.configured_max_tokens
        )

        self.client = self._client_for(provider)

    # ------------------------------------------------------------------ #
    # Request execution with failover
    # ------------------------------------------------------------------ #

    async def _execute(self, build_params, consume, **kwargs):
        """Run a request against the candidate chain.

        ``build_params(candidate)`` returns the request parameters for a
        candidate. ``consume(response)`` fully processes the response and
        returns the final result. Both run *after* the candidate has been
        activated, so model specific limits (such as the maximum number of
        output tokens) and provider specific message rules are applied.
        """
        await self._ensure_candidates()

        if self._legacy_client or not self._candidates:
            params = build_params(None)
            response = await self.client.chat.completions.create(**params, **kwargs)
            result = consume(response)
            if inspect.isawaitable(result):
                result = await result
            return result

        last_error: Optional[Exception] = None
        total = len(self._candidates)
        index = self._candidate_index
        tried = 0

        while tried < total:
            tried += 1
            candidate = self._candidates[index]
            self._activate(index)
            params = build_params(candidate)

            # A provider level problem (bad key, rate limit, outage) must skip
            # the remaining models of that provider, a model level problem
            # (rejected request, unusable answer) continues with the next
            # model of the same provider first.
            skip_provider = False

            attempt = 0
            while attempt < self.max_retries:
                attempt += 1
                try:
                    response = await self.client.chat.completions.create(
                        **params, **kwargs
                    )
                    result = consume(response)
                    if inspect.isawaitable(result):
                        result = await result
                    return result
                except OutputTruncated as exc:
                    last_error = exc
                    logger.warning(
                        f"Model '{candidate.model}' did not produce a usable "
                        f"answer ({exc}). Trying the next model."
                    )
                    break
                except ValueError:
                    raise
                except AuthenticationError as exc:
                    last_error = exc
                    skip_provider = True
                    logger.warning(
                        f"Provider '{candidate.provider.name}' rejected the API "
                        f"key ({exc.__class__.__name__}). Trying the next provider."
                    )
                    break
                except (BadRequestError, NotFoundError) as exc:
                    last_error = exc
                    logger.warning(
                        f"Model '{candidate.model}' on "
                        f"'{candidate.provider.name}' was rejected "
                        f"({exc.__class__.__name__}: {exc}). Trying the next model."
                    )
                    break
                except RateLimitError as exc:
                    last_error = exc
                    skip_provider = True
                    logger.warning(
                        f"Rate limited by '{candidate.provider.name}' on "
                        f"'{candidate.model}'. Trying the next provider."
                    )
                    break
                except (
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                ) as exc:
                    last_error = exc
                    if attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 10))
                        continue
                    skip_provider = True
                    logger.warning(
                        f"Provider '{candidate.provider.name}' is unreachable "
                        f"({exc.__class__.__name__}). Trying the next provider."
                    )
                    break
                except APIStatusError as exc:
                    last_error = exc
                    if exc.status_code >= 500 and attempt < self.max_retries:
                        await asyncio.sleep(min(2**attempt, 10))
                        continue
                    skip_provider = True
                    logger.warning(
                        f"Provider '{candidate.provider.name}' returned HTTP "
                        f"{exc.status_code}. Trying the next provider."
                    )
                    break
                except OpenAIError as exc:
                    last_error = exc
                    skip_provider = True
                    logger.warning(
                        f"Provider '{candidate.provider.name}' failed with "
                        f"{exc.__class__.__name__}. Trying the next provider."
                    )
                    break

            index = (
                self._next_provider_index(index)
                if skip_provider
                else (index + 1) % total
            )

        if last_error is None:  # pragma: no cover - defensive
            raise RuntimeError("No LLM candidate could serve the request.")
        raise last_error

    def _next_provider_index(self, index: int) -> int:
        """Index of the first candidate that belongs to a different provider."""
        current = self._candidates[index].provider.name
        total = len(self._candidates)
        for offset in range(1, total + 1):
            candidate_index = (index + offset) % total
            if self._candidates[candidate_index].provider.name != current:
                return candidate_index
        return (index + 1) % total

    def _validate_credentials(self) -> None:
        """Raise a clear error instead of an opaque 401 from the provider."""
        if not self.api_key:
            raise ValueError(
                f"No API key configured for model '{self.model}'. Set 'api_key' in "
                "config/config.toml (or use a ${ENV_VAR} placeholder / OPENMANUS_API_KEY)."
            )
        if not self.base_url:
            raise ValueError(
                f"No base_url configured for model '{self.model}'. Set 'base_url' in "
                "config/config.toml."
            )

    # ------------------------------------------------------------------ #
    # Token accounting
    # ------------------------------------------------------------------ #

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    # ------------------------------------------------------------------ #
    # Message helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    def _prepare_messages(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]],
        candidate: Optional[Candidate] = None,
    ) -> List[dict]:
        """Format the conversation and apply provider specific restrictions.

        ``messages[].name`` is rejected by Groq with a 400 error
        (https://console.groq.com/docs/openai). The field is optional in the
        OpenAI specification as long as ``tool_call_id`` is present, so it is
        removed by default for every provider and can be re-enabled with
        ``send_message_name = true``.
        """
        supports_images = self.supports_images
        if candidate is not None:
            configured = getattr(self.llm_config, "supports_images", None)
            supports_images = (
                configured
                if configured is not None
                else (
                    candidate.supports_images or model_supports_images(candidate.model)
                )
            )

        if system_msgs:
            formatted_system = self.format_messages(system_msgs, supports_images)
            formatted = formatted_system + self.format_messages(
                messages, supports_images
            )
        else:
            formatted = self.format_messages(messages, supports_images)

        if not self.send_message_name:
            for message in formatted:
                message.pop("name", None)

        return formatted

    def _build_params(
        self,
        candidate: Optional[Candidate],
        messages: List[dict],
        temperature=None,
        **extra,
    ):
        """Build the request parameters for a concrete model."""
        model = candidate.model if candidate is not None else self.model
        params: Dict = {"model": model, "messages": messages}

        if self.is_reasoning_model:
            params["max_completion_tokens"] = self.max_tokens
        else:
            params["max_tokens"] = self.max_tokens
            resolved_temperature = (
                temperature if temperature is not None else self.temperature
            )
            # Some models (reasoning models on OpenRouter, certain Gemini
            # builds) reject the temperature parameter outright.
            if resolved_temperature is not None:
                params["temperature"] = resolved_temperature

        params.update(extra)
        return params

    @staticmethod
    def _check_finish_reason(response) -> None:
        """Raise :class:`OutputTruncated` when the model produced no answer."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise OutputTruncated("the provider returned no choices")
        choice = choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message else None
        if content:
            return
        if finish_reason in ("length", "content_filter", "max_tokens"):
            raise OutputTruncated(
                f"the model stopped early (finish_reason={finish_reason})"
            )
        raise OutputTruncated("the provider returned an empty message")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            await self._ensure_candidates()
            self._validate_credentials()

            # Calculate input token count on the messages of the active model
            formatted = self._prepare_messages(messages, system_msgs)
            input_tokens = self.count_message_tokens(formatted)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            def build(candidate: Optional[Candidate]) -> Dict:
                payload = (
                    self._prepare_messages(messages, system_msgs, candidate)
                    if candidate is not None
                    else formatted
                )
                return self._build_params(candidate, payload, temperature=temperature)

            if not stream:
                # Non-streaming request
                usage_box: Dict = {}

                def consume(response) -> str:
                    usage_box["usage"] = getattr(response, "usage", None)
                    self._check_finish_reason(response)
                    return response.choices[0].message.content

                answer = await self._execute(build, consume, stream=False)

                # Update token counts
                usage = usage_box.get("usage")
                if usage is not None:
                    self.update_token_count(
                        getattr(usage, "prompt_tokens", 0) or 0,
                        getattr(usage, "completion_tokens", 0) or 0,
                    )

                return answer

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)

            async def consume_stream(response) -> str:
                collected_messages = []
                completion_text = ""
                async for chunk in response:
                    # Some providers (Groq, OpenAI with stream_options) emit a
                    # final chunk that only carries usage and has an empty
                    # choices list.
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    chunk_message = (
                        (getattr(delta, "content", None) or "") if delta else ""
                    )
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()
                if not full_response:
                    raise OutputTruncated("the model returned an empty stream")
                return full_response

            full_response = await self._execute(build, consume_stream, stream=True)

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(full_response)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            await self._ensure_candidates()
            self._validate_credentials()

            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images.
            # It is enough that *one* candidate of the chain can see images -
            # the failover may end up on that one.
            image_capable = self.supports_images or any(
                candidate.supports_images for candidate in self._candidates
            )
            if not image_capable:
                raise ValueError(
                    f"Model {self.model} does not support images. Set "
                    "supports_images = true in config.toml to force it on, or use a "
                    "vision capable model."
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            if not self.send_message_name:
                for message in all_messages:
                    message.pop("name", None)

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            def build(candidate: Optional[Candidate]) -> Dict:
                return self._build_params(
                    candidate,
                    all_messages,
                    temperature=temperature,
                    stream=stream,
                )

            # Handle non-streaming request
            if not stream:
                usage_box: Dict = {}

                def consume(response) -> str:
                    usage_box["usage"] = getattr(response, "usage", None)
                    self._check_finish_reason(response)
                    return response.choices[0].message.content

                answer = await self._execute(build, consume)

                usage = usage_box.get("usage")
                if usage is not None:
                    self.update_token_count(getattr(usage, "prompt_tokens", 0) or 0)
                return answer

            # Handle streaming request
            self.update_token_count(input_tokens)

            async def consume_stream(response) -> str:
                collected_messages = []
                completion_text = ""
                async for chunk in response:
                    # Providers may emit a trailing chunk that only carries usage.
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    chunk_message = (
                        (getattr(delta, "content", None) or "") if delta else ""
                    )
                    collected_messages.append(chunk_message)
                    completion_text += chunk_message
                    print(chunk_message, end="", flush=True)

                print()  # Newline after streaming
                full_response = "".join(collected_messages).strip()

                if not full_response:
                    raise OutputTruncated("the model returned an empty stream")

                return full_response

            return await self._execute(build, consume_stream)

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds
            tools: List of tools to use
            tool_choice: Tool choice strategy
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            ChatCompletionMessage: The model's response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If tools, tool_choice, or messages are invalid
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            await self._ensure_candidates()
            self._validate_credentials()

            # Validate tool_choice
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            formatted = self._prepare_messages(messages, system_msgs)

            # Calculate input token count
            input_tokens = self.count_message_tokens(formatted)

            # If there are tools, calculate token count for tool descriptions
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            # Validate tools if provided
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            def build(candidate: Optional[Candidate]) -> Dict:
                payload = (
                    self._prepare_messages(messages, system_msgs, candidate)
                    if candidate is not None
                    else formatted
                )
                params = self._build_params(candidate, payload, temperature=temperature)
                params["tools"] = tools
                params["tool_choice"] = tool_choice
                params["timeout"] = timeout
                params.update(kwargs)
                # Always use non-streaming for tool requests
                params["stream"] = False
                return params

            usage_box: Dict = {}

            def consume(response) -> ChatCompletionMessage:
                usage_box["usage"] = getattr(response, "usage", None)
                if not response.choices or not response.choices[0].message:
                    print(response)
                    return None
                message = response.choices[0].message
                # A model that only returns a tool call is fine; an empty
                # answer without any tool call is not.
                if not (getattr(message, "content", None) or message.tool_calls):
                    raise OutputTruncated(
                        "the model returned neither content nor a tool call"
                    )
                return message

            response = await self._execute(build, consume)

            if response is None:
                return None

            # Update token counts
            usage = usage_box.get("usage")
            if usage is not None:
                self.update_token_count(
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                )

            return response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_tool: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_tool: {e}")
            raise
