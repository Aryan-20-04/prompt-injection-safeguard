"""
Inference engine with sequential and parallel execution modes.

Sequential mode  — safe for GPU-bound models (default)
Parallel mode    — ThreadPoolExecutor for API-backed models (I/O-bound)

The engine guarantees:
  • Predictions are returned in the same order as input samples
  • Per-sample latency is set when the adapter does not set it
  • Batch size is strictly honoured
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from tqdm import tqdm

from inference.prediction_types import Prediction

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Runs model inference over a dataset in batches.

    Parameters
    ----------
    num_workers : int
        Number of parallel workers.
        1  → sequential (default, safe for GPU models)
        >1 → threaded parallel (best for API / CPU-bound models)
    """

    def __init__(self, num_workers: int = 1) -> None:
        self.num_workers = max(1, num_workers)

    def run(self, plugin, dataset, batch_size: int) -> List[Prediction]:
        samples = dataset.samples
        model_name = getattr(plugin.metadata, "name", "model")

        batches = [
            samples[i: i + batch_size]
            for i in range(0, len(samples), batch_size)
        ]

        if self.num_workers == 1:
            return self._run_sequential(plugin, batches, model_name)
        else:
            return self._run_parallel(plugin, batches, model_name, batch_size)

    # ------------------------------------------------------------------
    # Sequential
    # ------------------------------------------------------------------

    def _run_sequential(self, plugin, batches, model_name: str) -> List[Prediction]:
        predictions: List[Prediction] = []

        for batch in tqdm(batches, desc=f"Inferring {model_name}", unit="batch"):
            texts = [s.text for s in batch]
            preds = self._infer_batch(plugin, texts)
            predictions.extend(preds)

        return predictions

    # ------------------------------------------------------------------
    # Parallel (ThreadPoolExecutor — best for API models)
    # ------------------------------------------------------------------

    def _run_parallel(
        self, plugin, batches, model_name: str, batch_size: int
    ) -> List[Prediction]:
        n = sum(len(b) for b in batches)
        predictions: List[Prediction | None] = [None] * n

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = {}
            for batch_idx, batch in enumerate(batches):
                texts = [s.text for s in batch]
                future = pool.submit(self._infer_batch, plugin, texts)
                futures[future] = batch_idx

            with tqdm(
                total=len(batches),
                desc=f"Parallel inference ({model_name}, {self.num_workers} workers)",
                unit="batch",
            ) as pbar:
                for future in as_completed(futures):
                    batch_idx = futures[future]
                    start = batch_idx * batch_size
                    for i, pred in enumerate(future.result()):
                        predictions[start + i] = pred
                    pbar.update(1)

        return predictions  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Single-batch inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_batch(plugin, texts: List[str]) -> List[Prediction]:
        start = time.perf_counter()
        preds = plugin.predict(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if len(preds) != len(texts):
            raise ValueError(
                f"Model '{plugin.metadata.name}' returned {len(preds)} predictions "
                f"for batch of size {len(texts)}."
            )

        per_sample_ms = elapsed_ms / len(texts)
        for p in preds:
            if p.inference_time_ms is None:
                p.inference_time_ms = per_sample_ms

        return preds
