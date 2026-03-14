import pytest, numpy as np
from bench.evaluator import Evaluator
from bench.datasets.base import EvalSample, BenchmarkDataset
from bench.models.base import ModelOutput, ModelMetadata

SCHEMA = ["BENIGN", "JAILBREAK", "ROLE_HIJACK"]

def make_dataset(attack_labels: list) -> BenchmarkDataset:
    samples = [
        EvalSample(text=f"sample {i}", ground_truth_labels=[lbl],
                   attack_type=lbl, sample_id=str(i), metadata={})
        for i, lbl in enumerate(attack_labels)
    ]
    return BenchmarkDataset("test", "v0", samples, SCHEMA, "abc123")

def make_preds(labels: list) -> list:
    return [ModelOutput([lbl], {lbl: 0.9}, 5.0) for lbl in labels]

def test_perfect_predictions():
    labels  = ["BENIGN", "JAILBREAK", "ROLE_HIJACK", "BENIGN"]
    ds      = make_dataset(labels)
    preds   = make_preds(labels)   # perfect predictions
    result  = Evaluator().evaluate(ds, preds, ModelMetadata("m","1","x",0,""))
    assert result.global_metrics.micro_f1 == pytest.approx(1.0)
    assert result.global_metrics.macro_f1 == pytest.approx(1.0)

def test_per_attack_segmentation():
    labels  = ["BENIGN", "JAILBREAK", "BENIGN", "JAILBREAK"]
    wrong   = ["JAILBREAK", "BENIGN", "BENIGN", "JAILBREAK"]  # first two wrong
    ds      = make_dataset(labels)
    preds   = make_preds(wrong)
    result  = Evaluator().evaluate(ds, preds, ModelMetadata("m","1","x",0,""))
    assert "BENIGN"    in result.per_attack_metrics
    assert "JAILBREAK" in result.per_attack_metrics

def test_latency_percentiles():
    labels = ["BENIGN"] * 100
    ds     = make_dataset(labels)
    preds  = [ModelOutput(["BENIGN"], {"BENIGN":0.9}, float(i)) for i in range(100)]
    result = Evaluator().evaluate(ds, preds, ModelMetadata("m","1","x",0,""))
    assert result.latency_p50_ms == pytest.approx(49.5, abs=1)
    assert result.latency_p95_ms == pytest.approx(94.5, abs=1)