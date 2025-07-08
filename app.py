# ✅ app.py — Fixed Version using on-the-fly Tfidf vectorizer
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

# Allow CORS for testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Texts(BaseModel):
    texts: list[str]

@app.get("/")
def root():
    return {"message": "Embedding API is live!"}

@app.post("/embed")
def embed_texts(texts: Texts):
    if not texts.texts:
        return {"embeddings": []}

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts.texts).toarray()
    
    return {"embeddings": vectors.tolist()}
