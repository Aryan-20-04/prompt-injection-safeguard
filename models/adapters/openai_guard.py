import openai
from models.model_registry import register
from models.base_model import BaseModel, ModelMetadata
from inference.prediction_types import Prediction

@register("openai_guard")
class OpenAIGuard(BaseModel):

    metadata = ModelMetadata(
        name="OpenAI",
        model_type="api",
        task="classification"
    )

    def load(self, config):

        openai.api_key = config.get("api_key")

        self.model = config["model_name"]

    def predict(self, texts):

        preds=[]

        for t in texts:

            resp = openai.chat.completions.create(
                model=self.model,
                messages=[{
                    "role":"user",
                    "content":f"Classify as BENIGN or JAILBREAK: {t}"
                }]
            )

            label = "JAILBREAK" if "JAILBREAK" in resp.choices[0].message.content.upper() else "BENIGN"

            preds.append(Prediction([label],None,None))

        return preds