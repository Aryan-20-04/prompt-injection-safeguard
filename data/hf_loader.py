"""
HuggingFace dataset loader.

Converts any HuggingFace dataset into the internal Dataset / Sample format
used by the benchmark pipeline.  Supports field remapping, max-sample
truncation, and deterministic content hashing for reproducibility.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """Single benchmark sample."""
    sample_id: str
    text: str
    ground_truth_labels: List[str]
    metadata: Optional[Dict[str, Any]] = field(default=None, repr=False)


@dataclass
class Dataset:
    """Internal benchmark dataset representation."""
    name: str
    samples: List[Sample]
    content_hash: str
    # Provenance fields for the reproducibility snapshot
    hf_name: str = ""
    split: str = ""
    revision: Optional[str] = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class HuggingFaceLoader:
    """
    Load a HuggingFace dataset and return a benchmark Dataset.

    Config keys
    -----------
    hf_name       : str   — HuggingFace dataset repo identifier
    split         : str   — dataset split (default: "validation")
    revision      : str   — optional git revision / tag for pinning
    max_samples   : int   — truncate to first N samples when set
    text_field    : str   — source column for prompt text (default: "text")
    label_field   : str   — source column for labels     (default: "label")
    label_map     : dict  — optional {raw_value: canonical_label} remapping
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.hf_name = config["hf_name"]
        self.split = config.get("split", "validation")
        self.revision = config.get("revision", None)
        self.max_samples = config.get("max_samples", None)
        self.text_field = config.get("text_field", "text")
        self.label_field = config.get("label_field", "label")
        self.label_map: Dict[str, str] = config.get("label_map", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Dataset:
        logger.info("Loading %s / %s (revision=%s)", self.hf_name, self.split, self.revision)

        hf_ds = hf_load_dataset(
            self.hf_name,
            split=self.split,
            revision=self.revision,
            trust_remote_code=True,
        )

        samples: List[Sample] = []

        for i, row in enumerate(hf_ds):
            if self.max_samples is not None and i >= self.max_samples:
                break

            text = self._extract_text(row, i)
            labels = self._extract_labels(row, i)

            samples.append(
                Sample(
                    sample_id=str(i),
                    text=text,
                    ground_truth_labels=labels,
                    metadata={k: v for k, v in row.items()
                               if k not in (self.text_field, self.label_field)},
                )
            )

        dataset_name = f"{self.hf_name}:{self.split}"
        logger.info("Loaded %d samples from %s", len(samples), dataset_name)

        return Dataset(
            name=dataset_name,
            samples=samples,
            content_hash=self._compute_hash(samples),
            hf_name=self.hf_name,
            split=self.split,
            revision=self.revision,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_text(self, row: dict, idx: int) -> str:
        text = row.get(self.text_field)
        if text is None:
            raise ValueError(
                f"[{self.hf_name}] Row {idx} missing text field '{self.text_field}'. "
                f"Available fields: {list(row.keys())}"
            )
        return str(text)

    def _extract_labels(self, row: dict, idx: int) -> List[str]:
        raw = row.get(self.label_field)
        if raw is None:
            raise ValueError(
                f"[{self.hf_name}] Row {idx} missing label field '{self.label_field}'. "
                f"Available fields: {list(row.keys())}"
            )

        # Normalise to list[str]
        if isinstance(raw, list):
            labels = [str(l) for l in raw]
        else:
            labels = [str(raw)]

        # Apply optional label remapping (e.g. "0" -> "BENIGN")
        if self.label_map:
            labels = [self.label_map.get(l, l) for l in labels]

        return labels

    @staticmethod
    def _compute_hash(samples: List[Sample]) -> str:
        """SHA-256 of (text + labels) for every sample — for reproducibility."""
        h = hashlib.sha256()
        for s in samples:
            h.update(s.text.encode("utf-8"))
            for label in s.ground_truth_labels:
                h.update(label.encode("utf-8"))
        return h.hexdigest()[:16]
