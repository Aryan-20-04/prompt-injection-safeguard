METRIC_REGISTRY = {}


def register_metric(name):
    """
    Decorator to register new evaluation metrics.
    """

    def decorator(fn):

        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' already registered")

        METRIC_REGISTRY[name] = fn

        return fn

    return decorator


def get_metric(name):

    if name not in METRIC_REGISTRY:
        raise ValueError(f"Metric '{name}' not registered")

    return METRIC_REGISTRY[name]


def list_metrics():
    return list(METRIC_REGISTRY.keys())