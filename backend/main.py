from fastapi import FastAPI
from pydantic import BaseModel
import os
from gpt4all import GPT4All

# Path to GPT4ALL models
MODEL_PATH = "C:/Users/fores/gpt4all/models/"
MODEL_NAME = "mistral-7b-openorca.Q4_0.gguf"  

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        model = GPT4All(MODEL_NAME, model_path=MODEL_PATH, device="gpu") # Force GPU usage
        with model.chat_session():
            response = "".join(model.generate(
                prompt=f"You are Skog, a rogue warrior AI with a personality inspired by Norse warriors.\nUser: {request.message}\nSkog:",
                max_tokens=200,
                temp=0.7, # Adjusting temperature for better responses
                top_k=40, # Improve response quality
                streaming=True # Faster generation
            ))
        return {"response": response}

    except Exception as e:
        return {"error": str(e)}