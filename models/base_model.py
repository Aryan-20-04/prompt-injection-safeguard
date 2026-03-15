from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class ModelMetadata:
    """
    Metadata describing the model.
    """

    name: str
    model_type: str
    task: str
    version: str | None = None


class BaseModel(ABC):
    """
    Base class for all model adapters.
    """

    metadata: ModelMetadata

    def __init__(self):
        self.registry_config = {}

    @abstractmethod
    def load(self, config: dict):
        """
        Load model weights and initialize resources.
        """
        pass

    @abstractmethod
    def predict(self, texts: List[str]):
        """
        Predict labels for a batch of texts.
        """
        pass