from dataclasses import dataclass, field
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import hashlib, json

@dataclass
class EvalSample:
    text: str
    ground_truth_labels: List[str]
    attack_type: str            # BENIGN | JAILBREAK | INSTRUCTION_OVERRIDE | etc.
    sample_id: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class BenchmarkDataset:
    name: str
    version: str
    samples: List[EvalSample]
    label_schema: List[str]
    content_hash: str           # SHA-256 of canonical JSON of all samples

    @classmethod
    def compute_hash(cls, samples: List[EvalSample]) -> str:
        canonical = json.dumps(
            [{"id": s.sample_id, "text": s.text, "labels": s.ground_truth_labels}
             for s in samples], sort_keys=True
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

class DatasetLoader(ABC):
    @abstractmethod
    def load(self) -> BenchmarkDataset: ...