"""
Config loader and validator for benchmark.yaml.

Supports both:
  • datasets: [...]    — explicit dataset list
  • suite: <name>      — named suite (expands via data/suites.py)

One of the two must be present.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:

    REQUIRED_TOP_LEVEL = ["run", "models", "labels"]

    def load(self, config_path: str | Path) -> Dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        self._validate(config)
        return config

    def _validate(self, config: dict) -> None:
        for key in self.REQUIRED_TOP_LEVEL:
            if key not in config:
                raise ValueError(f"Missing required config section: '{key}'")

        if "output_dir" not in config["run"]:
            raise ValueError("run.output_dir is required")

        # Must have either suite or datasets
        has_suite    = "suite" in config
        has_datasets = "datasets" in config and isinstance(config["datasets"], list)
        if not has_suite and not has_datasets:
            raise ValueError(
                "Config must specify either 'suite: <name>' or 'datasets: [...]'"
            )

        if not isinstance(config["models"], list):
            raise ValueError("'models' must be a list")

        if not isinstance(config["labels"], list) or len(config["labels"]) == 0:
            raise ValueError("'labels' must be a non-empty list")

        # Validate each model entry has an 'id'
        for i, m in enumerate(config["models"]):
            if "id" not in m:
                raise ValueError(f"models[{i}] is missing required field 'id'")
