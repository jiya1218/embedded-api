from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load the transformer pipeline
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

try:
    embedder = pipeline("feature-extraction", model=MODEL_NAME, tokenizer=MODEL_NAME, trust_remote_code=True)
except Exception as e:
    raise RuntimeError(f"Model load failed: {e}")

# Define input schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Embedding API is running."}

@app.post("/embed")
def embed(input: TextInput):
    try:
        embedding = embedder(input.text)
        return {"embedding": embedding[0]}  # Use first layer output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
