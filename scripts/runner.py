# bench/runner.py — evaluation only, no Trainer.train()
class BenchmarkRunner:
    def run(self, config: BenchmarkConfig) -> RunSummary:
        """Load pre-trained weights and evaluate. Never trains."""
        snapshot = self._capture_snapshot(config)
        datasets  = [registry.load_dataset(d) for d in config.datasets]
        models    = [registry.load_model(m)   for m in config.models]

        results = []
        for model in models:
            model.load(config.get_model_config(model.metadata.name))
            for dataset in datasets:
                preds  = self._run_inference(model, dataset, config)
                result = self.evaluator.evaluate(dataset, preds, model.metadata)
                results.append(result)
        self.writer.write_all(results, snapshot, config.output_dir)
        return RunSummary(results=results, snapshot=snapshot)

    def _run_inference(self, model, dataset, config):
        batches = [dataset.samples[i:i+config.batch_size]
                   for i in range(0, len(dataset.samples), config.batch_size)]
        predictions = []
        for batch in tqdm(batches, desc=f"Inferring {model.metadata.name}"):
            texts = [s.text for s in batch]
            predictions.extend(model.predict(texts))
        return predictions