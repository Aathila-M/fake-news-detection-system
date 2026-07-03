import joblib
from src.preprocessing import clean_text

model = joblib.load("models/ml_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def predict_news(news_text):
    cleaned_text = clean_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])

    prediction = model.predict(vectorized_text)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(vectorized_text)[0]) * 100
    else:
        confidence = 90.0

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2),
        "cleaned_text": cleaned_text
    }