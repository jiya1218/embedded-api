from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = FastAPI()

vectorizer = TfidfVectorizer()

class EmbeddingRequest(BaseModel):
    texts: list[str]

@app.get("/")
def home():
    return {"message": "Embedding API is live!"}

@app.post("/embedding")
def generate_embeddings(req: EmbeddingRequest):
    try:
        vectors = vectorizer.fit_transform(req.texts)
        embeddings = vectors.toarray().tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))