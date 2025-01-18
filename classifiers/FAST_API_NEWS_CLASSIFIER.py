import os
from fastapi import FastAPI
import spacy
import pickle
from pydantic import BaseModel

class NewsRequest(BaseModel):
    text: str

app = FastAPI()
nlp = spacy.load("en_core_web_sm")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "news_classifier.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model, label_mapping = pickle.load(f)

def preprocess(text):
    doc = nlp(text)
    words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(words)

@app.post("/predict/")
async def predict(news: NewsRequest):
    processed_text = preprocess(news.text)
    prediction = model.predict([processed_text])    
    predicted_label = {v: k for k, v in label_mapping.items()}[prediction[0]]
    return {"prediction": predicted_label}
