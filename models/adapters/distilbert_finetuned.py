"""
DistilBERT fine-tuned for prompt injection detection.

Model: fmops/distilbert-prompt-injection
The fastest fine-tuned encoder for this task.
66M parameters — runs on CPU if needed.

Labels: INJECTION / SAFE → remapped to JAILBREAK / BENIGN
"""

from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin


@register("distilbert_finetuned")
class DistilBERTFinetunedAdapter(HFClassifierMixin):
    """
    DistilBERT fine-tuned specifically for prompt injection detection.

    Usage in benchmark.yaml:
        - id: distilbert_finetuned
          model_name: fmops/distilbert-prompt-injection
          max_len: 512
    """

    metadata = ModelMetadata(
        name="DistilBERT-PromptInjection",
        model_type="encoder",
        hf_id="fmops/distilbert-prompt-injection",
    )

    def load(self, config: dict) -> None:
        super().load(config)
        # Remap model's native labels to benchmark canonical labels
        _remap = {
            "INJECTION": "JAILBREAK",
            "SAFE": "BENIGN",
            "injection": "JAILBREAK",
            "safe": "BENIGN",
        }
        self.label_map = {
            k: _remap.get(v, v) for k, v in self.label_map.items()
        }