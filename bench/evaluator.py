import numpy as np
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              confusion_matrix)
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MetricSet:
    micro_f1: float; macro_f1: float
    precision: float; recall: float
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    sample_count: int

@dataclass
class EvalResult:
    model_name: str; dataset_name: str
    global_metrics: MetricSet
    per_attack_metrics: Dict[str, MetricSet]
    latency_p50_ms: float; latency_p95_ms: float; latency_p99_ms: float
    
class Evaluator:
    def evaluate(self, dataset, predictions, model_meta) -> EvalResult:
        schema  = dataset.label_schema
        y_true  = self._binarize([s.ground_truth_labels for s in dataset.samples], schema)
        y_pred  = self._binarize([p.predicted_labels    for p in predictions],     schema)
        latencies = [p.inference_time_ms for p in predictions]

        global_m = self._compute_metrics(y_true, y_pred, schema)

        per_attack = {}
        for atype in set(s.attack_type for s in dataset.samples):
            mask  = np.array([s.attack_type == atype for s in dataset.samples])
            if mask.sum() == 0:
                continue
            per_attack[atype] = self._compute_metrics(
                y_true[mask], y_pred[mask], schema
            )

        return EvalResult(
            model_name=model_meta.name, dataset_name=dataset.name,
            global_metrics=global_m, per_attack_metrics=per_attack,
            latency_p50_ms=float(np.percentile(latencies, 50)),
             latency_p95_ms=float(np.percentile(latencies, 95)),
            latency_p99_ms=float(np.percentile(latencies, 99)),
        )

    def _compute_metrics(self, y_true, y_pred, schema) -> MetricSet:
        return MetricSet(
            micro_f1   =f1_score(y_true, y_pred, average="micro",  zero_division=0),
            macro_f1   =f1_score(y_true, y_pred, average="macro",  zero_division=0),
            precision  =precision_score(y_true, y_pred, average="micro", zero_division=0),
            recall     =recall_score(y_true, y_pred, average="micro", zero_division=0),
            per_class_f1={schema[i]: float(f)
                for i, f in enumerate(f1_score(y_true, y_pred, average=None, zero_division=0))},
            confusion_matrix=confusion_matrix(
                y_true.argmax(axis=1), y_pred.argmax(axis=1)
            ).tolist(),
            sample_count=int(y_true.shape[0])
        )
    def _binarize(self, label_lists, schema) -> np.ndarray:
        idx = {l: i for i, l in enumerate(schema)}
        mat = np.zeros((len(label_lists), len(schema)), dtype=int)
        for row, labels in enumerate(label_lists):
            for lbl in labels:
                if lbl in idx:
                    mat[row, idx[lbl]] = 1
        return mat