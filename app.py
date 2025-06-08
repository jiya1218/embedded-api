from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
import numpy as np

app = FastAPI()

vectorizer = TfidfVectorizer()

class EmbeddingRequest(BaseModel):
    texts: List[str]

@app.get("/")
def root():
    return {"message": "Embedding API is live!"}

@app.post("/embed")
def embed_texts(request: EmbeddingRequest):
    texts = request.texts
    if not texts:
        return {"embeddings": []}

    vectors = vectorizer.fit_transform(texts)
    embeddings = vectors.toarray().tolist()
    return {"embeddings": embeddings}
