from pymongo import MongoClient
from datetime import datetime, timezone


class MongoLogger:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27018/")
        self.db = self.client["llm_monitoring"]
        self.collection = self.db["call_logs"]

    def log_call(self, log_data):
        log_data["timestamp"] = datetime.now(timezone.utc)
        self.collection.insert_one(log_data)

    def close(self):
        self.client.close()