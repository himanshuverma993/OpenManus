"""Helpers for resolving the user prompt in CLI and notebook environments."""

import os
import sys


def interactive_stdin_available() -> bool:
    """Return True when stdin is an interactive terminal."""
    try:
        if sys.stdin is None or sys.stdin.closed:
            return False
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def resolve_prompt(
    cli_prompt: str | None = None,
    *,
    env_var: str = "OPENMANUS_PROMPT",
    allow_interactive: bool = True,
    prompt_text: str = "Enter your prompt: ",
) -> str | None:
    """Resolve a prompt from (in order) the CLI argument, the environment and stdin.

    Returns ``None`` when no prompt could be resolved, e.g. inside a notebook
    where no interactive stdin exists.
    """
    if cli_prompt and cli_prompt.strip():
        return cli_prompt.strip()

    env_prompt = os.environ.get(env_var, "").strip()
    if env_prompt:
        return env_prompt

    if allow_interactive and interactive_stdin_available():
        try:
            return input(prompt_text).strip() or None
        except (EOFError, OSError):
            return None

    return None
