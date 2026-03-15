import importlib
import pkgutil
def discover_models():
    import models.adapters
    package = models.adapters
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        importlib.import_module(f"models.adapters.{module_name}")