import os
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the port Render sets; default to 10000 if not provided
PORT = int(os.getenv("PORT", 10000))

app = FastAPI(title="TF‑IDF Embedding API")

# We'll re‑fit a TfidfVectorizer on each incoming text (single document).
# That means: if your text has N unique tokens, you'll get an N‑dimensional vector.
# For a simple “demo” chatbot embedding, this is fine and guaranteed to use <512 MB.
vectorizer = TfidfVectorizer()

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def health_check():
    """
    Render’s health checker pings “GET /”. By returning {"status":"ok"},
    Render sees a 200 response immediately and marks the service healthy.
    """
    return {"status": "ok"}

@app.post("/embed")
async def embed_text(req: TextRequest):
    """
    1. We take the incoming JSON: {"text": "some string"}.
    2. Fit TF‑IDF on that single document => yields a 1×D sparse vector.
    3. Convert to a dense array of length D and return as a list of floats.
    """
    # Fit on the single document (so vocabulary = the tokens in req.text)
    tfidf_matrix = vectorizer.fit_transform([req.text])
    # Convert sparse→dense; shape = (1, D) => [0] to get the 1D array
    vec = tfidf_matrix.toarray()[0]
    return {"embedding": vec.tolist()}

if __name__ == "__main__":
    # If you run “python app.py” locally, this block will fire.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
