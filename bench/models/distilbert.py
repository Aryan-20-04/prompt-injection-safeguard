import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import ModelPlugin, ModelOutput, ModelMetadata
from .registry import register

@register("distilbert-injection-detector")
class DistilBERTInjectionDetector(ModelPlugin):

    def load(self, config: dict) -> None:
        self.labels   = config["labels"]
        self.max_len  = config.get("max_len", 256)
        self.device   = config.get("device", "cpu")
        model_name    = config.get("model_name", "distilbert-base-uncased")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.labels),
            problem_type="multi_label_classification"
        ).to(self.device)
        self.model.eval()
    def predict(self, texts: list) -> list:
        outputs = []
        for text in texts:
            t0 = time.perf_counter()

            enc = self.tokenizer(
                text, truncation=True, padding="max_length",
                max_length=self.max_len, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits[0].cpu().numpy()

            probs    = 1 / (1 + np.exp(-logits))   # sigmoid
            preds    = [self.labels[i] for i, p in enumerate(probs) if p > 0.5]
            prob_map = {lbl: float(p) for lbl, p in zip(self.labels, probs)}

            elapsed_ms = (time.perf_counter() - t0) * 1000
            outputs.append(ModelOutput(preds, prob_map, elapsed_ms))
        return outputs
    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="distilbert-injection-detector", version="1.0.0",
            architecture="DistilBERT-base-uncased",
            parameter_count=66_000_000,
            model_url="https://huggingface.co/distilbert-base-uncased"
        )