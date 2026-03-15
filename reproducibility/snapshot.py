"""
Reproducibility snapshot — captures full environment state for a run.

Saved to run_metadata.json alongside every benchmark run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def capture_snapshot(config: dict, seed: int = 42) -> Dict[str, Any]:
    """
    Build a reproducibility snapshot dict and set all RNG seeds.

    Returns a dict that is serialised to run_metadata.json.
    """
    _set_seeds(seed)

    import torch

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_hash(),
        "git_dirty": _is_dirty(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda or "none",
        "gpu": _gpu_name(),
        "gpu_memory_gb": _gpu_memory_gb(),
        "cpu": platform.processor(),
        "os": platform.platform(),
        "hostname": platform.node(),
        "seed": seed,
        "config_hash": _hash_config(config),
        # filled in after datasets are loaded:
        "dataset_hashes": {},
        # filled in after models are evaluated:
        "model_versions": {},
    }

    # Capture installed packages only when not in a CI environment
    # (pip freeze is slow; skip it with CI=1)
    if not os.environ.get("CI"):
        snapshot["packages"] = _pip_freeze()

    return snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug("RNG seeds set to %d", seed)


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
    except Exception:
        return "no-git"


def _is_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"


def _gpu_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / 1e9, 2)
    except Exception:
        pass
    return 0.0


def _pip_freeze() -> list:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return [p for p in result.stdout.splitlines() if p.strip()]


def _hash_config(config: dict) -> str:
    serialised = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()[:12]
