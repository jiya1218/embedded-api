import os
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import numpy as np
import uvicorn

# 1. Read the PORT that Render sets (default to 10000 if missing)
PORT = int(os.getenv("PORT", 10000))

app = FastAPI()

# 2. Load a small transformer model & tokenizer (CPU‐only)
#    We use "sentence-transformers/all-MiniLM-L6-v2" as an example
#    because it is reasonably small and works without torch.
#    Under the hood, transformers will automatically pick backends to compute embeddings.

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


class TextRequest(BaseModel):
    text: str


@app.get("/")
async def health_check():
    """
    A simple health‐check endpoint. Render will ping this to see if the service is up.
    """
    return {"status": "ok"}


@app.post("/embed")
async def embed_text(req: TextRequest):
    """
    1. Tokenize input text
    2. Run through huggingface AutoModel
    3. Take the [CLS] token embedding as a fixed‐length vector
    4. Return that vector (as a Python list of floats)
    """
    inputs = tokenizer(
        req.text,
        return_tensors="np",
        padding=True,
        truncation=True,
        max_length=128,
    )

    # The model returns a dictionary with 'last_hidden_state'
    outputs = model(**inputs)
    # Shape of last_hidden_state: (batch_size=1, seq_len, hidden_size)
    # We simply take the first token ([CLS]) embedding:
    cls_embedding = outputs.last_hidden_state[:, 0, :].reshape(-1)  # shape (hidden_size,)

    # Convert to Python list
    embedding_list = cls_embedding.tolist()
    return {"embedding": embedding_list}


if __name__ == "__main__":
    # This block is only used if you run `python app.py` manually
    uvicorn.run(app, host="0.0.0.0", port=PORT)
