# ✅ app.py for real embeddings using TfidfVectorizer (no HuggingFace, no PyTorch)
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Enable CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Texts(BaseModel):
    texts: list[str]

vectorizer = TfidfVectorizer()
vectorizer.fit(["sample sentence for initializing tfidf vectorizer"])

@app.get("/")
def root():
    return {"message": "Embedding API is live!"}

@app.post("/embed")
def embed_texts(texts: Texts):
    vectors = vectorizer.transform(texts.texts).toarray()
    return {"embeddings": vectors.tolist()}
