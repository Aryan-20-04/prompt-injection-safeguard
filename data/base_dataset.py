from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Sample:
    """
    Represents a single dataset example.
    """
    sample_id: str
    text: str
    ground_truth_labels: List[str]
    metadata: Dict[str, Any] | None = None


@dataclass
class Dataset:
    """
    Standard dataset format used across the benchmark.
    """
    name: str
    samples: List[Sample]
    content_hash: str

    def size(self) -> int:
        return len(self.samples)

    def label_set(self) -> List[str]:
        labels = set()
        for s in self.samples:
            labels.update(s.ground_truth_labels)
        return sorted(list(labels))