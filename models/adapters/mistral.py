from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.llm_base import LLMClassifierMixin


@register("mistral")
class MistralAdapter(LLMClassifierMixin):
    """
    Mistral instruct adapter.
    Set model_name in benchmark.yaml, e.g.:
      mistralai/Mistral-7B-Instruct-v0.3
    """
    metadata = ModelMetadata(
        name="Mistral",
        model_type="llm",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
    )
