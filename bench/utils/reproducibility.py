# bench/utils/reproducibility.py
import subprocess, sys, json, time, hashlib
from pathlib import Path

def capture_snapshot(config: dict, seed: int = 42) -> dict:
    import torch
    _set_seeds(seed)
    return {
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit":  _git_hash(),
        "python":      sys.version,
        "torch":       torch.__version__,
        "cuda":        torch.version.cuda or "none",
        "gpu":         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "packages":    _pip_freeze(),
        "seed":        seed,
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True)
                                      .encode()).hexdigest()[:12],
    }

def _set_seeds(seed: int):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()[:12]
    except Exception:
        return "no-git"

def _pip_freeze() -> list:
    result = subprocess.run(["pip", "freeze"], capture_output=True, text=True)
    return result.stdout.strip().split("\n")