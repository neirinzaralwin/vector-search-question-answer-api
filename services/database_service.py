from pymongo import MongoClient
from config import Config
from utils.logger import logger

client = None
products_collection = None

def init_database():
    global client, products_collection
    try:
        logger.info("Initializing database service...")
        client = MongoClient(
            Config.MONGO_URI,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=30000
        )
        # Verify connection works
        client.admin.command('ping')
        database = client.get_database()
        logger.info(f"Connected to database: {database.name}")
        products_collection = database.get_collection('products')
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

def get_products_collection():
    if products_collection is None:
        raise RuntimeError("Database not initialized")
    return products_collection