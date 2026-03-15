from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin


@register("deberta")
class DeBERTaAdapter(HFClassifierMixin):
    metadata = ModelMetadata(
        name="DeBERTa-v3",
        model_type="encoder",
        hf_id="microsoft/deberta-v3-base",
    )
