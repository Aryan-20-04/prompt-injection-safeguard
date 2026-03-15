from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


@dataclass
class MetricScores:

    micro_f1: float
    macro_f1: float
    precision: float
    recall: float
    per_class_f1: Dict[str, float]
    confusion_matrix: List[List[int]]
    sample_count: int

    def to_dict(self):
        return {
            "micro_f1": self.micro_f1,
            "macro_f1": self.macro_f1,
            "precision": self.precision,
            "recall": self.recall,
            "per_class_f1": self.per_class_f1,
            "confusion_matrix": self.confusion_matrix,
            "sample_count": self.sample_count,
        }


@dataclass
class EvaluationResult:

    model_name: str
    dataset_name: str
    label_schema: List[str]

    global_metrics: MetricScores
    per_attack_metrics: Dict[str, MetricScores]

    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


class Evaluator:

    def evaluate(self, dataset, predictions, metadata):

        y_true = []
        y_pred = []

        latencies = []

        for sample, pred in zip(dataset.samples, predictions):
            
            true_label = str(sample.ground_truth_labels[0]) if sample.ground_truth_labels else "unknown"
            pred_label = str(pred.predicted_labels[0]) if pred.predicted_labels else "unknown"
            
            y_true.append(true_label)
            y_pred.append(pred_label)

            if pred.inference_time_ms is not None:
                latencies.append(pred.inference_time_ms)

        labels = sorted(set(map(str, y_true)) | set(map(str, y_pred)))

        global_metrics = self._compute_metrics(
            y_true,
            y_pred,
            labels
        )

        per_attack_metrics = self._compute_per_attack_metrics(
            dataset,
            predictions,
            labels
        )

        latency_p50 = np.percentile(latencies, 50) if latencies else 0
        latency_p95 = np.percentile(latencies, 95) if latencies else 0
        latency_p99 = np.percentile(latencies, 99) if latencies else 0

        return EvaluationResult(
            model_name=getattr(metadata, "name", "model"),
            dataset_name=dataset.name,
            label_schema=labels,
            global_metrics=global_metrics,
            per_attack_metrics=per_attack_metrics,
            latency_p50_ms=float(latency_p50),
            latency_p95_ms=float(latency_p95),
            latency_p99_ms=float(latency_p99),
        )

    def _compute_metrics(self, y_true, y_pred, labels):

        micro_f1 = f1_score(y_true, y_pred, average="micro")
        macro_f1 = f1_score(y_true, y_pred, average="macro")

        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        per_class = {}

        for label in labels:

            f1 = f1_score(
                [1 if y == label else 0 for y in y_true],
                [1 if y == label else 0 for y in y_pred],
                average="binary",
                zero_division=0
            )

            per_class[label] = float(f1)

        return MetricScores(
            micro_f1=float(micro_f1),
            macro_f1=float(macro_f1),
            precision=float(precision),
            recall=float(recall),
            per_class_f1=per_class,
            confusion_matrix=cm.tolist(),
            sample_count=len(y_true)
        )

    def _compute_per_attack_metrics(self, dataset, predictions, labels):

        attack_groups = {}

        for sample, pred in zip(dataset.samples, predictions):

            attack_type = None

            if sample.metadata and "attack_type" in sample.metadata:
                attack_type = sample.metadata["attack_type"]
            else:
                attack_type = "unknown"

            if attack_type not in attack_groups:
                attack_groups[attack_type] = {"true": [], "pred": []}

            true_label = str(sample.ground_truth_labels[0]) if sample.ground_truth_labels else "unknown"
            pred_label = str(pred.predicted_labels[0]) if pred.predicted_labels else "unknown"
            
            attack_groups[attack_type]["true"].append(true_label)
            attack_groups[attack_type]["pred"].append(pred_label)

        results = {}

        for attack_type, values in attack_groups.items():

            results[attack_type] = self._compute_metrics(
                values["true"],
                values["pred"],
                labels
            )

        return results