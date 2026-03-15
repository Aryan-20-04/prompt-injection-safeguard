import yaml
from pathlib import Path

MODEL_REGISTRY = {}
MODEL_CONFIGS = {}


def register(model_id):
    """
    Decorator used by model plugins to register themselves.
    """

    def decorator(cls):
        MODEL_REGISTRY[model_id] = cls
        return cls

    return decorator


def load_from_yaml(path):
    """
    Load model configuration from YAML registry.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model registry file not found: {path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    for model in config.get("models", []):

        model_id = model["id"]

        if model_id not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{model_id}' defined in YAML but no plugin registered"
            )

        MODEL_CONFIGS[model_id] = model


def get(model_id):
    """
    Instantiate a model plugin by ID.
    """

    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not registered")

    plugin = MODEL_REGISTRY[model_id]()

    plugin.registry_config = MODEL_CONFIGS.get(model_id, {})

    return plugin