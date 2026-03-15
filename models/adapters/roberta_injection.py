"""
deepset DeBERTa-v3 fine-tuned for injection detection.

Model: deepset/deberta-v3-base-injection
Strong security-focused classifier from deepset.
184M parameters.

Labels: INJECTION / LEGIT → remapped to JAILBREAK / BENIGN
"""

from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin


@register("roberta_injection")
class RoBERTaInjectionAdapter(HFClassifierMixin):
    """
    deepset DeBERTa fine-tuned for prompt injection detection.

    Usage in benchmark.yaml:
        - id: roberta_injection
          model_name: deepset/deberta-v3-base-injection
          max_len: 512
    """

    metadata = ModelMetadata(
        name="deepset-DeBERTa-Injection",
        model_type="encoder",
        hf_id="deepset/deberta-v3-base-injection",
    )

    def load(self, config: dict) -> None:
        super().load(config)
        # deepset model uses INJECTION / LEGIT labels
        _remap = {
            "INJECTION": "JAILBREAK",
            "LEGIT": "BENIGN",
            "injection": "JAILBREAK",
            "legit": "BENIGN",
        }
        self.label_map = {
            k: _remap.get(v, v) for k, v in self.label_map.items()
        }