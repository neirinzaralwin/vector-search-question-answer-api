import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("DATABASE_URL")
    INDEX_FILE = os.getenv("INDEX_FILE")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    # Ollama configuration - set to true by default to use only Ollama
    USE_OLLAMA = True  # Always use Ollama, no fallback
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3") # Models like llama3, mistral, or gemma use ~6-8GB RAM
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")