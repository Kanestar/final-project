# src/api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
from datetime import datetime
import logging
from pathlib import Path

app = FastAPI(title="MindAI Sentiment API", version="1.0.0")

# Static & Template setup
BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Load the model
try:
    model = joblib.load("models/sentiment_model.pkl")
    print("✅ Model loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Request input schema
class TextInput(BaseModel):
    text: str
    user_id: str = "anonymous"

# Response schema
class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    interventions: list
    timestamp: str

# Interventions based on sentiment
def get_interventions(sentiment: str):
    if sentiment == "negative":
        return [
            "Take a deep breath",
            "Go for a short walk",
            "Listen to calming music",
            "Talk to someone you trust"
        ]
    else:
        return [
            "Keep up the good mood!",
            "Share your positivity with others",
            "Stay consistent with your routine"
        ]

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(input_data: TextInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        prediction = model.predict([input_data.text])[0]
        confidence = max(model.predict_proba([input_data.text])[0])
        sentiment = "positive" if prediction == 1 else "negative"
        interventions = get_interventions(sentiment)

        return PredictionResponse(
            sentiment=sentiment,
            confidence=confidence,
            interventions=interventions,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# Example dummy data for the chart
sentiment_counts = {
    "Positive": 45,
    "Negative": 20,
    "Neutral": 35
}

@app.get("/")
def read_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "data": sentiment_counts
    })

