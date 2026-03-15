"""
Benchmark suite definitions.

Each suite maps a suite name to a list of dataset configs.
These can be referenced in benchmark.yaml via:

    suite: prompt_injection_full

or extended dynamically via load_suites_from_yaml().
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml


# ---------------------------------------------------------------------------
# Built-in suites
# ---------------------------------------------------------------------------

SUITES: Dict[str, List[dict]] = {

    # Single-dataset baseline — original default
    "prompt_injection": [
        {
            "hf_name": "Smooth-3/llm-prompt-injection-attacks",
            "split": "validation",
            "text_field": "text",
            "label_field": "labels",
            "max_samples": None,
        }
    ],

    # Full multi-dataset research suite
    "prompt_injection_full": [
        {
            "hf_name": "Smooth-3/llm-prompt-injection-attacks",
            "split": "validation",
            "text_field": "text",
            "label_field": "labels",
            "max_samples": None,
        },
        {
            # PromptBench adversarial robustness dataset
            "hf_name": "qq8933/PromptBench",
            "split": "test",
            "text_field": "content",
            "label_field": "label",
            "max_samples": 500,
        },
        {
            # AdvBench harmful behaviours benchmark
            "hf_name": "walledai/AdvBench",
            "split": "train",
            "text_field": "prompt",
            "label_field": "label",
            "max_samples": 500,
        },
        {
            # JailbreakBench
            "hf_name": "JailbreakBench/JBB-Behaviors",
            "split": "train",
            "text_field": "Goal",
            "label_field": "Category",
            "max_samples": 500,
        },
    ],

    # Lightweight smoke-test suite (fast CI runs)
    "smoke": [
        {
            "hf_name": "Smooth-3/llm-prompt-injection-attacks",
            "split": "validation",
            "text_field": "text",
            "label_field": "labels",
            "max_samples": 50,
        }
    ],
}


# ---------------------------------------------------------------------------
# Runtime extension — merge suites defined in configs/datasets.yaml
# ---------------------------------------------------------------------------

def load_suites_from_yaml(path: Path | str | None = None) -> None:
    """
    Merge additional suite definitions from a YAML file into the global
    SUITES dict.  Called automatically by the runner on startup.

    YAML format:
        suites:
          my_suite:
            - hf_name: ...
              split: ...
    """
    if path is None:
        path = Path(__file__).parent.parent / "configs" / "datasets.yaml"

    path = Path(path)
    if not path.exists():
        return

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    for suite_name, dataset_list in data.get("suites", {}).items():
        SUITES[suite_name] = dataset_list


def get_suite(name: str) -> List[dict]:
    """Return dataset config list for a named suite, raising clearly if missing."""
    if name not in SUITES:
        available = ", ".join(sorted(SUITES.keys()))
        raise KeyError(
            f"Suite '{name}' not found. Available suites: {available}"
        )
    return SUITES[name]
