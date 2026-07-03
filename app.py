from flask import Flask, render_template, request, jsonify

from src.prediction import predict_news
from src.database import create_table, save_prediction, get_all_predictions

app = Flask(__name__)

create_table()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form.get("news_text")

    result = predict_news(news_text)

    save_prediction(
        news_text,
        result["prediction"],
        result["confidence"]
    )

    return render_template(
        "result.html",
        news_text=news_text,
        prediction=result["prediction"],
        confidence=result["confidence"]
    )


@app.route("/dashboard")
def dashboard():
    records = get_all_predictions()
    return render_template("dashboard.html", records=records)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    news_text = data.get("news_text", "")

    result = predict_news(news_text)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)