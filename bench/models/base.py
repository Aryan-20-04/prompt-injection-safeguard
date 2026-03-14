from dataclasses import dataclass
from typing import List, Dict
from abc import ABC, abstractmethod
import time

@dataclass
class ModelOutput:
    predicted_labels: List[str]
    label_probabilities: Dict[str, float]
    inference_time_ms: float       # Measured INSIDE predict(), not by the runner

@dataclass
class ModelMetadata:
    name: str
    version: str
    architecture: str
    parameter_count: int
    model_url: str
    description: str = ""

class ModelPlugin(ABC):

    @abstractmethod
    def load(self, config: dict) -> None:
        """Initialize model weights. Called once before any predict() calls."""
        ...

    @abstractmethod
    def predict(self, texts: List[str]) -> List[ModelOutput]:
        """
        Run inference on a batch of texts.
        MUST return len(texts) outputs in the same order.
        MUST measure inference_time_ms per sample.
        """
        ...

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Return model card info for leaderboard display."""
        ...