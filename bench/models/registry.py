# bench/models/registry.py
import importlib
from typing import Dict, Type
import yaml
from .base import ModelPlugin

_REGISTRY: Dict[str, Type[ModelPlugin]] = {}

def register(model_id: str):
    """Decorator: @register('roberta-injection-detector')"""
    def decorator(cls):
        _REGISTRY[model_id] = cls
        return cls
    return decorator

def load_from_yaml(registry_path: str) -> Dict[str, Type[ModelPlugin]]:
    """Load all plugins declared in model_registry.yaml at runtime."""
    with open(registry_path) as f:
        data = yaml.safe_load(f)
    for entry in data.get("models", []):
        module_path, cls_name = entry["class"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls    = getattr(module, cls_name)
        _REGISTRY[entry["id"]] = cls
    return _REGISTRY

def get(model_id: str) -> ModelPlugin:
    if model_id not in _REGISTRY:
        raise KeyError(f"Model '{model_id}' not found. Run: python scripts/validate_plugin.py")
    return _REGISTRY[model_id]()