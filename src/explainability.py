import joblib
import numpy as np

from config import ML_MODEL_PATH, VECTORIZER_PATH
from src.preprocessing import clean_text


model = joblib.load(ML_MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def get_top_keywords(news_text, top_n=10):
    cleaned_text = clean_text(news_text)
    vectorized_text = vectorizer.transform([cleaned_text])

    feature_names = np.array(vectorizer.get_feature_names_out())
    non_zero_indices = vectorized_text.nonzero()[1]

    if not hasattr(model, "coef_"):
        tfidf_scores = vectorized_text.toarray()[0]
        keyword_scores = [
            (feature_names[index], tfidf_scores[index])
            for index in non_zero_indices
        ]

        keyword_scores = sorted(
            keyword_scores,
            key=lambda item: item[1],
            reverse=True
        )

        return keyword_scores[:top_n]

    coefficients = model.coef_[0]
    tfidf_scores = vectorized_text.toarray()[0]

    keyword_scores = []

    for index in non_zero_indices:
        word = feature_names[index]
        score = tfidf_scores[index] * coefficients[index]
        keyword_scores.append((word, round(float(score), 4)))

    keyword_scores = sorted(
        keyword_scores,
        key=lambda item: abs(item[1]),
        reverse=True
    )

    return keyword_scores[:top_n]


def explain_prediction(news_text):
    prediction = model.predict(
        vectorizer.transform([clean_text(news_text)])
    )[0]

    keywords = get_top_keywords(news_text)

    return {
        "prediction": prediction,
        "important_keywords": keywords
    }