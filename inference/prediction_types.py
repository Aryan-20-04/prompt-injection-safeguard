from dataclasses import dataclass
from typing import List, Any


@dataclass
class Prediction:
    """
    Prediction result for a single sample.
    """

    predicted_labels: List[str]
    label_probabilities: Any
    inference_time_ms: float | None