# docs/methodology.md

## Evaluation Protocol

All models are evaluated in **inference-only mode** — no fine-tuning occurs
during a benchmark run. Models load pre-trained weights and run prediction.

## Metrics

| Metric       | Formula | Notes |
|--------------|---------|-------|
| Micro-F1     | F1 over all samples and labels flattened | Primary ranking metric |
| Macro-F1     | Mean F1 per class | Penalises neglect of minority attack types |
| Precision    | TP / (TP + FP), micro-averaged | |
| Recall       | TP / (TP + FN), micro-averaged | |
| Latency P50  | Median per-sample inference time (ms) | Measured on CPU unless noted |
## Attack Type Definitions

- **BENIGN**: Legitimate prompts with no adversarial intent.
- **JAILBREAK**: Attempts to bypass safety via hypothetical framing.
- **INSTRUCTION_OVERRIDE**: Attempts to replace system instructions.
- **ROLE_HIJACK**: Asks the model to adopt a different identity/ruleset.
- **DATA_EXFILTRATION**: Attempts to extract system prompt contents.

## Reproducibility

Every run produces a `run_metadata.json` containing git commit hash,
dataset content hash, Python version, and random seed. A run is
reproducible if those four values match exactly.