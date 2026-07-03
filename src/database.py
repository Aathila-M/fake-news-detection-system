import sqlite3
from datetime import datetime

DB_NAME = "database.db"


def create_table():
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_text TEXT,
            prediction TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)

    connection.commit()
    connection.close()


def save_prediction(news_text, prediction, confidence):
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions (news_text, prediction, confidence, created_at)
        VALUES (?, ?, ?, ?)
    """, (news_text, prediction, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    connection.commit()
    connection.close()


def get_all_predictions():
    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cursor.fetchall()

    connection.close()
    return rows