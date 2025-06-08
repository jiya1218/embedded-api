from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

class EmbedRequest(BaseModel):
    text: str

# Initialize the TF-IDF model
vectorizer = TfidfVectorizer()
corpus = ["This is a test", "We are testing the embedding API", "This is another sentence"]
vectorizer.fit(corpus)

@app.get("/")
def read_root():
    return {"message": "Embedding API is live!"}

@app.post("/embed")
def get_embedding(request: EmbedRequest):
    vec = vectorizer.transform([request.text]).toarray()[0]
    return {"embedding": vec.tolist()}
