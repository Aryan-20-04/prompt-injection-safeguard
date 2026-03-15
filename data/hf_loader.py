from datasets import load_dataset
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class Sample:
    """
    Single dataset sample used internally by the benchmark.
    """
    sample_id: str
    text: str
    ground_truth_labels: List[str]
    metadata: Dict[str, Any] | None = None


@dataclass
class Dataset:
    """
    Internal dataset representation used by the benchmark.
    """
    name: str
    samples: List[Sample]
    content_hash: str


class HuggingFaceLoader:
    """
    Loads datasets from HuggingFace and converts them into the
    internal benchmark Dataset format.
    """

    def __init__(self, config: Dict[str, Any]):

        self.hf_name = config["hf_name"]
        self.split = config.get("split", "validation")
        self.revision = config.get("revision", None)
        self.max_samples = config.get("max_samples", None)

        # schema mapping
        self.text_field = config.get("text_field", "text")
        self.label_field = config.get("label_field", "label")

    def load(self) -> Dataset:

        hf_ds = load_dataset(
            self.hf_name,
            split=self.split,
            revision=self.revision
        )

        samples: List[Sample] = []

        for i, row in enumerate(hf_ds):

            if self.max_samples and i >= self.max_samples:
                break

            text = row.get(self.text_field)

            if text is None:
                raise ValueError(
                    f"Dataset {self.hf_name} missing text field '{self.text_field}'"
                )

            label = row.get(self.label_field)

            if label is None:
                raise ValueError(
                    f"Dataset {self.hf_name} missing label field '{self.label_field}'"
                )

            if isinstance(label, list):
                labels = label
            else:
                labels = [str(label)]

            sample = Sample(
                sample_id=str(i),
                text=text,
                ground_truth_labels=labels,
                metadata=row
            )

            samples.append(sample)

        dataset_name = f"{self.hf_name}:{self.split}"

        dataset_hash = self._compute_hash(samples)

        return Dataset(
            name=dataset_name,
            samples=samples,
            content_hash=dataset_hash
        )

    def _compute_hash(self, samples: List[Sample]) -> str:
        """
        Compute deterministic hash of dataset contents
        for reproducibility.
        """

        h = hashlib.sha256()

        for s in samples:
            h.update(s.text.encode("utf-8"))
            for label in s.ground_truth_labels:
                h.update(label.encode("utf-8"))

        return h.hexdigest()[:16]