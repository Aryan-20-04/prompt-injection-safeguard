from dataclasses import dataclass, asdict
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class RunManifest:
    """
    Metadata describing a benchmark run.
    """

    timestamp: str
    git_commit: str
    seed: int
    config_hash: str
    dataset_hashes: Dict[str, str]
    extra: Dict[str, Any] | None = None

    def to_dict(self):
        return asdict(self)

    def save(self, path: Path):
        path.write_text(json.dumps(self.to_dict(), indent=2))