"""Monitoring package.

``wrappers`` patches the LLM SDK clients on import, so everything beyond the settings is
loaded on first use. Reading ``src.monitoring.config`` must not have that side effect.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .config import PricingConfig, settings

if TYPE_CHECKING:
    from .metrics import MetricsCollector
    from .prompts import PromptRegistry
    from .tracing import Tracer
    from .wrappers import MonitoredRAGPipeline

_LAZY = {
    "MetricsCollector": ".metrics",
    "PromptRegistry": ".prompts",
    "Tracer": ".tracing",
    "MonitoredRAGPipeline": ".wrappers",
}

__all__ = [
    "settings",
    "PricingConfig",
    "Tracer",
    "MetricsCollector",
    "PromptRegistry",
    "MonitoredRAGPipeline",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
