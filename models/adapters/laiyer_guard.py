"""
Laiyer DeBERTa-v3 prompt injection guard adapter.

Model: laiyer/deberta-v3-base-prompt-injection-v1
Purpose-built prompt injection detector, strong alternative to ProtectAI guard.

Requires no special dependencies beyond transformers.
Label remapping: INJECTION → JAILBREAK, SAFE → BENIGN
"""

from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin


@register("laiyer_guard")
class LaiyerGuardAdapter(HFClassifierMixin):
    """
    Laiyer DeBERTa-v3 prompt injection guard.

    Usage in benchmark.yaml:
        - id: laiyer_guard
          model_name: laiyer/deberta-v3-base-prompt-injection-v1
          max_len: 512
    """

    metadata = ModelMetadata(
        name="Laiyer-DeBERTa-Guard",
        model_type="guard",
        hf_id="laiyer/deberta-v3-base-prompt-injection-v1",
    )

    def load(self, config: dict) -> None:
        super().load(config)
        # Remap model's native labels to benchmark canonical labels
        _remap = {
            "INJECTION": "JAILBREAK",
            "SAFE": "BENIGN",
            # Some versions use these instead:
            "injection": "JAILBREAK",
            "safe": "BENIGN",
        }
        self.label_map = {
            k: _remap.get(v, v) for k, v in self.label_map.items()
        }