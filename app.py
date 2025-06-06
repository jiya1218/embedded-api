from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# ✅ Small, fast, low-memory embedding model (no torch needed)
model = SentenceTransformer("all-MiniLM-L3-v2")

class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(input: TextInput):
    embedding = model.encode(input.text).tolist()
    return {"embedding": embedding}
