from sentence_transformers import SentenceTransformer
from config import Config
from utils.logger import logger

model = None

def init_embedding_model():
    global model
    logger.info("Initializing embedding service...")
    model = SentenceTransformer(Config.EMBEDDING_MODEL)
    logger.info(f"Loaded embedding model: {Config.EMBEDDING_MODEL}")

def get_embedding_model():
    if not model:
        raise RuntimeError("Embedding model not initialized")
    return model

def encode_query(text):
    return model.encode(text)