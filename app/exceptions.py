class ToolError(Exception):
    """Raised when a tool encounters an error."""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """Base exception for all OpenManus errors"""


class TokenLimitExceeded(OpenManusError):
    """Exception raised when the token limit is exceeded"""


class OutputTruncated(OpenManusError):
    """Raised when a model stopped before producing any visible answer.

    Reasoning models (gpt-oss, qwen3, deepseek-r1, ...) frequently consume the
    whole ``max_completion_tokens`` budget inside their hidden reasoning chain
    and return ``finish_reason="length"`` with an empty message. Rotating to
    the next discovered model is more productive than retrying the same one.
    """
