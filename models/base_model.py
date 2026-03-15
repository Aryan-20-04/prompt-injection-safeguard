"""
Base class and metadata type for all model adapters.

Every adapter must:
  1. Subclass BaseModel
  2. Set class-level `metadata: ModelMetadata`
  3. Implement load(config) and predict(texts)
  4. Be decorated with @register("<id>")

The adapter will then be auto-discovered by models/discovery.py and
made available to the benchmark runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from inference.prediction_types import Prediction


@dataclass
class ModelMetadata:
    """
    Descriptive metadata attached to every model adapter.

    Fields
    ------
    name        : Human-readable display name
    model_type  : "encoder" | "guard" | "llm" | "api"
    task        : "classification" (only supported task for now)
    version     : Optional model version string for tracking
    hf_id       : Optional HuggingFace model card identifier
    """
    name: str
    model_type: str          # encoder | guard | llm | api
    task: str = "classification"
    version: Optional[str] = None
    hf_id: Optional[str] = None


class BaseModel(ABC):
    """
    Base class for all benchmark model adapters.

    Subclasses must declare a class-level `metadata` attribute of type
    ModelMetadata and implement the two abstract methods below.
    """

    metadata: ModelMetadata  # declared at class level by each subclass

    def __init__(self) -> None:
        # Populated by the registry after instantiation
        self.registry_config: dict = {}

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, config: dict) -> None:
        """
        Load model weights and initialise any resources.

        Parameters
        ----------
        config : dict
            Keys available to all adapters:
              device      — "cpu" | "cuda" | "mps"
              model_name  — model checkpoint / identifier string
              labels      — list of canonical label strings
              max_len     — tokenizer max sequence length
            API adapters also receive:
              api_key     — credential string (injected from env if not set)
        """

    @abstractmethod
    def predict(self, texts: List[str]) -> List[Prediction]:
        """
        Run inference on a batch of texts.

        Parameters
        ----------
        texts : list[str]
            A batch of raw prompt strings.

        Returns
        -------
        list[Prediction]
            Must be the same length as `texts`.
            Each Prediction should set predicted_labels and
            label_probabilities (or None for generative models).
        """
