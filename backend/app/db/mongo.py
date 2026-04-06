from pymongo import MongoClient

from app.core.config import settings

client = MongoClient(settings.mongodb_uri)
db = client[settings.mongodb_db]


def get_collection(name: str):
    return db[name]
