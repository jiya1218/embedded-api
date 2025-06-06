from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, pipeline
import numpy as np
from sklearn.preprocessing import normalize

app = FastAPI()

# Load tokenizer and feature-extraction pipeline
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
embedder = pipeline("feature-extraction", model=MODEL_NAME, tokenizer=tokenizer)

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Embedding API is running"}

@app.post("/embed")
def generate_embedding(input: TextInput):
    try:
        # Run feature extraction
        features = embedder(input.text)

        # Pool token-level vectors into sentence-level vector (mean pooling)
        embeddings = np.mean(features[0], axis=0)

        # Normalize the vector
        normalized = normalize([embeddings])[0].tolist()
        return {"embedding": normalized}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))