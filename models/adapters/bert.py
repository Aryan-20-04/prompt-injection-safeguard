from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin


@register("bert")
class BERTAdapter(HFClassifierMixin):
    metadata = ModelMetadata(
        name="BERT",
        model_type="encoder",
        hf_id="bert-base-uncased",
    )
