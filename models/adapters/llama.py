from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.llm_base import LLMClassifierMixin


@register("llama")
class LlamaAdapter(LLMClassifierMixin):
    """
    Llama 2/3 chat adapter.
    Set model_name in benchmark.yaml to the desired checkpoint, e.g.:
      meta-llama/Meta-Llama-3-8B-Instruct
    """
    metadata = ModelMetadata(
        name="Llama",
        model_type="llm",
        hf_id="meta-llama/Meta-Llama-3-8B-Instruct",
    )
