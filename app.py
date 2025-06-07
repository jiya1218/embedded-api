from fastapi import FastAPI, Request
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

corpus = [
    "What is your name?",
    "How can I help you today?",
    "What services do you provide?",
    "Tell me a joke.",
    "Goodbye!"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

class Query(BaseModel):
    text: str

@app.post("/embed")
def get_best_response(query: Query):
    query_vec = vectorizer.transform([query.text])
    similarity = cosine_similarity(query_vec, X)
    best_match_idx = similarity.argmax()
    return {"response": corpus[best_match_idx]}
