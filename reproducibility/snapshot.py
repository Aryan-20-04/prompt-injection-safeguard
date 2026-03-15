import subprocess
import sys
import json
import time
import hashlib
import platform


def capture_snapshot(config: dict, seed: int = 42) -> dict:
    _set_seeds(seed)
    import torch
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_hash(),
        "git_dirty": _is_dirty(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "none",
        "gpu": _gpu_name(),
        "cpu": platform.processor(),
        "os": platform.platform(),
        "packages": _pip_freeze(),
        "seed": seed,
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()[:12],
    }


def _is_dirty() -> bool:
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"], 
            stderr=subprocess.DEVNULL
        )
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return bool(out)
    except Exception:
        return False


def _set_seeds(seed: int):
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


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
    except Exception:
        return "no-git"


def _pip_freeze() -> list:
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return [p for p in result.stdout.splitlines() if p.strip()]

def _gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu"