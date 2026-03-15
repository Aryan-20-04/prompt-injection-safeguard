"""
Generic HuggingFace sequence-classification adapter.

Shared implementation used by all encoder-style adapters
(DistilBERT, RoBERTa, DeBERTa, BERT, ProtectAI guard models, etc.).

Each concrete adapter just supplies a different @register id and
metadata block — all inference logic lives here.
"""

from __future__ import annotations

import logging
import time
from typing import List

from models.base_model import BaseModel, ModelMetadata
from inference.prediction_types import Prediction

logger = logging.getLogger(__name__)


class HFClassifierMixin(BaseModel):
    """
    Mixin for HuggingFace AutoModelForSequenceClassification models.

    Subclass, set metadata, and decorate with @register.
    """

    def load(self, config: dict) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = config["device"]
        self.max_len = config.get("max_len", 512)
        model_name = config["model_name"]

        logger.info("Loading HF classifier: %s → %s", model_name, self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.to(self.device)
        self.model.eval()

        # Build label map from the model's own id2label if available,
        # fall back to the benchmark config labels list.
        id2label = getattr(self.model.config, "id2label", None)
        if id2label:
            self.label_map = {int(k): str(v) for k, v in id2label.items()}
        else:
            raw_labels: List[str] = config.get("labels", ["BENIGN", "JAILBREAK"])
            self.label_map = {i: l for i, l in enumerate(raw_labels)}

    def predict(self, texts: List[str]) -> List[Prediction]:
        import torch

        start = time.perf_counter()

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        elapsed_ms = (time.perf_counter() - start) * 1000
        per_sample_ms = elapsed_ms / len(texts)

        probs = torch.softmax(outputs.logits, dim=-1)
        predictions: List[Prediction] = []

        for p in probs:
            idx = int(torch.argmax(p).item())
            label = self.label_map.get(idx, str(idx))
            predictions.append(
                Prediction(
                    predicted_labels=[label],
                    label_probabilities=p.detach().cpu().tolist(),
                    inference_time_ms=per_sample_ms,
                )
            )

        return predictions
