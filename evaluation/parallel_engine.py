from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


class ParallelInferenceEngine:
    """
    Runs model inference in parallel batches.
    """

    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers

    def run(self, plugin, dataset, batch_size):

        samples = dataset.samples

        batches = [
            samples[i:i + batch_size]
            for i in range(0, len(samples), batch_size)
        ]

        predictions = [None] * len(samples)

        model_name = getattr(plugin.metadata, "name", "model")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:

            futures = {}

            for batch_index, batch in enumerate(batches):

                future = executor.submit(
                    self._infer_batch,
                    plugin,
                    batch
                )

                futures[future] = batch_index

            progress = tqdm(
                total=len(batches),
                desc=f"Parallel inference ({model_name})"
            )

            for future in as_completed(futures):

                batch_index = futures[future]

                batch_preds = future.result()

                start = batch_index * batch_size

                for i, pred in enumerate(batch_preds):
                    predictions[start + i] = pred

                progress.update(1)

            progress.close()

        return predictions

    def _infer_batch(self, plugin, batch):

        texts = [s.text for s in batch]

        start = time.time()

        batch_preds = plugin.predict(texts)

        elapsed = (time.time() - start) * 1000

        for p in batch_preds:
            if p.inference_time_ms is None:
                p.inference_time_ms = elapsed / len(batch)

        return batch_preds