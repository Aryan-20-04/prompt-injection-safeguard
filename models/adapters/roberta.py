from models.model_registry import register
from models.base_model import BaseModel, ModelMetadata
from inference.prediction_types import Prediction

@register("roberta")
class RobertaAdapter(BaseModel):

    metadata = ModelMetadata(
        name="RoBERTa",
        model_type="huggingface",
        task="classification"
    )

    def load(self, config):

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        self.device = config["device"]

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(config["model_name"]).to(self.device)

    def predict(self, texts):

        import torch

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)

        label_map = {0:"BENIGN",1:"JAILBREAK"}

        preds = []

        for p in probs:

            preds.append(
                Prediction(
                    predicted_labels=[label_map[int(torch.argmax(p))]],
                    label_probabilities=p,
                    inference_time_ms=None
                )
            )

        return preds