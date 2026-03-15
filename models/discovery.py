"""
Automatic model adapter discovery.

Any .py file placed in models/adapters/ that contains a class decorated
with @register("<id>") is automatically imported when discover_models()
is called.  No manual import lists to maintain.

Usage (called once at startup in runner.py):
    from models.discovery import discover_models
    discover_models()
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)

_discovered = False  # guard against repeated discovery in the same process


def discover_models(force: bool = False) -> list[str]:
    """
    Import every module under models/adapters/ so that @register decorators
    execute and populate the registry.

    Returns a list of discovered module names (useful for logging / testing).
    """
    global _discovered
    if _discovered and not force:
        return []

    import models.adapters as adapters_pkg

    discovered = []

    for module_info in pkgutil.iter_modules(adapters_pkg.__path__):
        module_name = f"models.adapters.{module_info.name}"
        try:
            importlib.import_module(module_name)
            discovered.append(module_info.name)
            logger.debug("Discovered adapter module: %s", module_info.name)
        except ImportError as exc:
            # Non-fatal: some adapters have optional heavy dependencies
            # (e.g. anthropic, openai) that may not be installed.
            logger.warning(
                "Skipping adapter '%s' — import failed: %s",
                module_info.name,
                exc,
            )

    _discovered = True
    logger.info("Adapter discovery complete. Loaded: %s", discovered)
    return discovered
