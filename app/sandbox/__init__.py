"""
Docker Sandbox Module

Provides secure containerized execution environment with resource limits
and isolation for running untrusted code.

The Docker specific classes are imported lazily so that the ``docker`` package
is only required when the sandbox is actually used. This keeps lightweight
deployments (e.g. a text agent on Kaggle) free of an unnecessary dependency.
"""
from typing import Any

from app.sandbox.client import (
    BaseSandboxClient,
    LocalSandboxClient,
    create_sandbox_client,
)
from app.sandbox.core.exceptions import (
    SandboxError,
    SandboxResourceError,
    SandboxTimeoutError,
)


__all__ = [
    "DockerSandbox",
    "SandboxManager",
    "BaseSandboxClient",
    "LocalSandboxClient",
    "create_sandbox_client",
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxResourceError",
]

_LAZY_IMPORTS = {
    "DockerSandbox": "app.sandbox.core.sandbox",
    "SandboxManager": "app.sandbox.core.manager",
}


def __getattr__(name: str) -> Any:
    """Resolve Docker dependent classes on first use (PEP 562)."""
    module_path = _LAZY_IMPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
