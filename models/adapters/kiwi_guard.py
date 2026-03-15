"""
ProtectAI DeBERTa-v3 prompt injection guard model adapter.

Model: protectai/deberta-v3-base-prompt-injection
This is a purpose-built guard model with its own id2label that maps
directly to INJECTION / SAFE, which we remap to JAILBREAK / BENIGN.
"""

from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin


@register("kiwi_guard")
class KiwiGuardAdapter(HFClassifierMixin):
    metadata = ModelMetadata(
        name="ProtectAI-DeBERTa-Guard",
        model_type="guard",
        hf_id="protectai/deberta-v3-base-prompt-injection",
    )

    def load(self, config: dict) -> None:
        super().load(config)
        # Normalise guard model's native labels to benchmark canonical labels
        _remap = {"INJECTION": "JAILBREAK", "SAFE": "BENIGN"}
        self.label_map = {
            k: _remap.get(v, v) for k, v in self.label_map.items()
        }
