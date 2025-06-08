# app.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

class EmbeddingRequest(BaseModel):
    texts: list[str]

vectorizer = TfidfVectorizer()

@app.post("/embed")
def embed(req: EmbeddingRequest):
    embeddings = vectorizer.fit_transform(req.texts).toarray()
    return {"embeddings": embeddings.tolist()}

@app.get("/")
def root():
    return {"message": "Embedding API is working!"}