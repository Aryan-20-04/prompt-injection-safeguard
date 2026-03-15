import time
from tqdm import tqdm


class InferenceEngine:

    def run(self, plugin, dataset, batch_size):

        samples = dataset.samples
        predictions = []

        model_name = getattr(plugin.metadata, "name", "model")

        for i in tqdm(
            range(0, len(samples), batch_size),
            desc=f"Inferring {model_name}",
            unit="batch"
        ):

            batch = samples[i:i + batch_size]

            texts = [s.text for s in batch]

            start = time.time()

            batch_preds = plugin.predict(texts)

            elapsed = (time.time() - start) * 1000

            if len(batch_preds) != len(batch):
                raise ValueError(
                    f"Model returned {len(batch_preds)} predictions "
                    f"for batch size {len(batch)}"
                )

            for p in batch_preds:
                if p.inference_time_ms is None:
                    p.inference_time_ms = elapsed / len(batch)

            predictions.extend(batch_preds)

        return predictions