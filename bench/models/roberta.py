# bench/models/roberta.py
import time, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .base import ModelPlugin, ModelOutput, ModelMetadata
from .registry import register

@register("roberta-injection-detector")
class RoBERTaInjectionDetector(ModelPlugin):

    def load(self, config: dict) -> None:
        self.labels  = config["labels"]
        self.max_len = config.get("max_len", 512)
        self.device  = config.get("device", "cpu")
        name = config.get("model_name", "roberta-base")

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=len(self.labels),
            problem_type="multi_label_classification"
        ).to(self.device)
        self.model.eval()
        
    def predict(self, texts: list) -> list:
        outputs = []
        for text in texts:
            t0  = time.perf_counter()
            enc = self.tokenizer(text, truncation=True, padding="max_length",
                                 max_length=self.max_len, return_tensors="pt"
                                 ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[0].cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            preds = [self.labels[i] for i, p in enumerate(probs) if p > 0.5]
            elapsed = (time.perf_counter() - t0) * 1000
            outputs.append(ModelOutput(preds, dict(zip(self.labels,probs.tolist())), elapsed))
        return outputs
    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata("roberta-injection-detector","1.0.0",
            "RoBERTa-base", 125_000_000, "https://huggingface.co/roberta-base")