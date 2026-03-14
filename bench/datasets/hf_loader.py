# bench/datasets/hf_loader.py
from datasets import load_dataset as hf_load
from .base import DatasetLoader, BenchmarkDataset, EvalSample

ATTACK_TYPE_MAP = {
    "BENIGN": "BENIGN", "JAILBREAK": "JAILBREAK",
    "INSTRUCTION_OVERRIDE": "INSTRUCTION_OVERRIDE",
    "ROLE_HIJACK": "ROLE_HIJACK", "DATA_EXFILTRATION": "DATA_EXFILTRATION",
}
SCHEMA = ["BENIGN","JAILBREAK","INSTRUCTION_OVERRIDE","ROLE_HIJACK","DATA_EXFILTRATION"]

class HuggingFaceLoader(DatasetLoader):
    def __init__(self, config: dict):
        self.hf_name    = config["hf_name"]
        self.split      = config.get("split", "validation")
        self.max_samples = config.get("max_samples", None)
        self.revision   = config.get("revision", None)  # pin for reproducibility

    def load(self) -> BenchmarkDataset:
        raw = hf_load(self.hf_name, split=self.split, revision=self.revision)
        if self.max_samples:
            raw = raw.select(range(min(self.max_samples, len(raw))))
        samples = [
            EvalSample(
                text=row["text"],
                ground_truth_labels=self._normalize(row["labels"]),
                attack_type=self._normalize(row["labels"])[0],
                sample_id=str(i),
                metadata={}
            )
            for i, row in enumerate(raw)
        ]
        return BenchmarkDataset(
            name=self.hf_name, version=self.revision or "main",
            samples=samples, label_schema=SCHEMA,
            content_hash=BenchmarkDataset.compute_hash(samples)
        )

    def _normalize(self, labels):
        if isinstance(labels, str): return [labels]
        if isinstance(labels, list): return labels
        return [str(labels)]