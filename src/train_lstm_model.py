import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

from config import (
    DATASET_PATH,
    MODEL_DIR,
    LSTM_MODEL_PATH,
    TOKENIZER_PATH,
    MAX_WORDS,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    TEST_SIZE,
    RANDOM_STATE,
    EPOCHS,
    BATCH_SIZE
)

from src.preprocessing import clean_text


def train_lstm_model():
    data = pd.read_csv(DATASET_PATH)

    data = data.dropna(subset=["text", "label"])
    data["text"] = data["text"].apply(clean_text)

    x = data["text"]
    y = data["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    x_train_pad = pad_sequences(
        x_train_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    x_test_pad = pad_sequences(
        x_test_seq,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    model = Sequential([
        Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        LSTM(32),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        x_train_pad,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping]
    )

    loss, accuracy = model.evaluate(x_test_pad, y_test)

    print("LSTM Test Accuracy:", accuracy)

    predictions = model.predict(x_test_pad)
    predicted_labels = (predictions > 0.5).astype(int)

    print(classification_report(y_test, predicted_labels))
    print(confusion_matrix(y_test, predicted_labels))

    MODEL_DIR.mkdir(exist_ok=True)

    model.save(LSTM_MODEL_PATH)

    with open(TOKENIZER_PATH, "wb") as file:
        pickle.dump(tokenizer, file)

    with open(MODEL_DIR / "label_encoder.pkl", "wb") as file:
        pickle.dump(label_encoder, file)

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("LSTM Training Accuracy")
    plt.savefig(MODEL_DIR / "lstm_accuracy.png")

    print("LSTM model, tokenizer, and label encoder saved successfully.")


if __name__ == "__main__":
    train_lstm_model()