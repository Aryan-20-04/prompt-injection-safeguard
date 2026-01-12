# Prompt Injection Detection using Transformers

This repository contains a complete training pipeline for a **multi-label prompt injection detection model** trained on a Hugging Face dataset.

## Features

- Multi-label classification of prompt attacks
- Detects:
  - BENIGN
  - JAILBREAK
  - INSTRUCTION_OVERRIDE
  - ROLE_HIJACK
  - DATA_EXFILTRATION
- Uses Hugging Face `Trainer`
- Supports GPU / Colab training
- Evaluation with Micro & Macro F1

## Model

- Backbone: DistilBERT
- Task: Multi-label text classification
- Loss: BCEWithLogitsLoss (via Trainer)

## Results

Achieved high performance on the validation set:

- Micro F1 ≈ 0.98
- Macro F1 ≈ 0.95

## Training

```bash
python train_prompt_injection_model.py
```
