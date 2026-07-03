import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import clean_text


data = pd.read_csv("dataset/news.csv")

data["text"] = data["text"].apply(clean_text)

x = data["text"]
y = data["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42)
}

best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    model.fit(x_train_vec, y_train)
    predictions = model.predict(x_test_vec)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel:", name)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_name = name

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/ml_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nBest Model:", best_name)
print("Best Accuracy:", best_accuracy)

cm = confusion_matrix(y_test, best_model.predict(x_test_vec))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("models/confusion_matrix.png")