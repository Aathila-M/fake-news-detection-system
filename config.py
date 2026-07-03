from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "news.csv"

MODEL_DIR = BASE_DIR / "models"
ML_MODEL_PATH = MODEL_DIR / "ml_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.h5"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.pkl"

DATABASE_PATH = BASE_DIR / "database.db"

MAX_WORDS = 10000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 128

TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 8
BATCH_SIZE = 32