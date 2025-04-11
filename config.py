import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("DATABASE_URL")
    INDEX_FILE = os.getenv("INDEX_FILE")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")