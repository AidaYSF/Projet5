from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import tensorflow_hub as hub
import os

os.environ["TFHUB_CACHE_DIR"] = "./tfhub_cache"

app = FastAPI()

# Charger le modèle SVC et le MultiLabelBinarizer
with open("model_SVC_USE/model.pkl", "rb") as f:
    svc_model = pickle.load(f)
with open("model_SVC_USE/mlb.pkl", "rb") as f:
    mlb = pickle.load(f)

# Charger Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict_tags(req: TextRequest):
    texts = [req.text]
    embeddings = use_model(tf.constant(texts)).numpy() # type: ignore
    y_pred = svc_model.predict(embeddings)
    tags = mlb.inverse_transform(y_pred)[0]
    return {"tags": tags}
