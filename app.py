# app.py — proper embedding API using SentenceTransformer

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Enable CORS (useful for frontend or API test tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Texts(BaseModel):
    texts: list[str]

# ✅ Load real model (384-dim vectors, very fast and semantic)
model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def root():
    return {"message": "🚀 Embedding API is live and using SentenceTransformer"}

@app.post("/embed")
def embed_texts(texts: Texts):
    embeddings = model.encode(texts.texts)
    return {"embeddings": embeddings.tolist()}
