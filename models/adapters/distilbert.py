from models.model_registry import register
from models.base_model import BaseModel, ModelMetadata
from inference.prediction_types import Prediction

@register("distilbert")
class DistilBERTPlugin:

    metadata = {
        "name": "DistilBERT",
        "type": "huggingface",
        "task": "classification"
    }

    def load(self, config):

        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        model_name = config["model_name"]
        device = config["device"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        self.model.to(device)
        self.device = device

    def predict(self, texts):

        import torch

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = []

        for p in probs:
            label_map={0: "BENIGN", 1: "JAILBREAK"}

            predictions.append(
                Prediction(
                    predicted_labels=[label_map[int(torch.argmax(p).item())]],
                    label_probabilities=p,
                    inference_time_ms=None
                )
            )

        return predictions