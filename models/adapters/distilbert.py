from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin


@register("distilbert")
class DistilBERTAdapter(HFClassifierMixin):
    metadata = ModelMetadata(
        name="DistilBERT",
        model_type="encoder",
        hf_id="distilbert-base-uncased",
    )
