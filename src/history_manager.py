import sqlite3
from datetime import datetime
import json
import os

class HistoryManager:
    def __init__(self, db_path='../data/history.db'):
        self.db_path = db_path
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    drug_sequence TEXT,
                    protein_sequence TEXT,
                    interaction_score REAL,
                    drug_name TEXT,
                    prediction_details TEXT
                )
            ''')
    
    def save_prediction(self, drug_sequence, protein_sequence, score, drug_name, details):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO predictions 
                (timestamp, drug_sequence, protein_sequence, interaction_score, drug_name, prediction_details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), drug_sequence, protein_sequence, score, drug_name, json.dumps(details)))
    
    def get_history(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()