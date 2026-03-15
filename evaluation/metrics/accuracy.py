from evaluation.metric_registry import register_metric


@register_metric("accuracy")
def accuracy(y_true, y_pred):

    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)

    return correct / len(y_true)