# utils/case_db.py
import sqlite3
from datetime import datetime

class CaseStorage:
    def __init__(self, db_path="history/case.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS case_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                role TEXT,        -- 'user' or 'assistant'
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def save_message(self, session_id, user_id, role, message):
        self.conn.execute(
            "INSERT INTO case_records (session_id, user_id, role, message, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, role, message, datetime.utcnow())
        )
        self.conn.commit()

    def get_messages(self, user_id, session_id=None):
        cursor = self.conn.cursor()
        if session_id:
            cursor.execute(
                "SELECT role, message FROM case_records WHERE user_id=? AND session_id=? ORDER BY id",
                (user_id, session_id)
            )
        else:
            cursor.execute(
                "SELECT role, message FROM case_records WHERE user_id=? ORDER BY id",
                (user_id,)
            )
        return cursor.fetchall()
