def rank_models(results, metric="micro_f1"):
    """
    Rank models by a chosen metric.
    """

    sorted_results = sorted(
        results,
        key=lambda r: getattr(r.global_metrics, metric),
        reverse=True
    )

    ranked = []

    for i, r in enumerate(sorted_results, start=1):

        ranked.append({
            "rank": i,
            "model": r.model_name,
            "dataset": r.dataset_name,
            "micro_f1": r.global_metrics.micro_f1,
            "macro_f1": r.global_metrics.macro_f1,
            "precision": r.global_metrics.precision,
            "recall": r.global_metrics.recall
        })

    return ranked