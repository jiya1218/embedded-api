from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.preprocessing import normalize
import torch
import os

# Cache model files to avoid re-downloading
os.environ["TRANSFORMERS_CACHE"] = "/opt/render/project/.cache/huggingface"

app = FastAPI()

# Load tokenizer and model (small, no torch inference)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    normed = normalize(cls_embedding.numpy())
    return normed[0].tolist()

@app.get("/")
def root():
    return {"message": "Embedding API is running."}

@app.post("/embed")
async def embed(request: Request):
    data = await request.json()
    text = data.get("text", "")
    embedding = get_embedding(text)
    return {"embedding": embedding}
