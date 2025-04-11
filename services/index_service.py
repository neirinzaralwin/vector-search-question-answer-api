import hnswlib
import os
import re
from config import Config
from services.database_service import get_products_collection
from services.embedding_service import get_embedding_model
from utils.logger import logger
from datetime import datetime

index = None
product_ids = []
products_data = []

def preprocess_description(text):
    """
    Clean and standardize product descriptions for better embeddings
    Returns lowercase text with:
    - Special characters removed
    - Standardized cannabis terminology
    - Consistent number formatting
    - Reduced noise
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Standardize cannabis terms
    replacements = {
        r'thc:?\s*(\d+\.?\d*)%?': r'thc \1%',
        r'cbd:?\s*(\d+\.?\d*)%?': r'cbd \1%',
        r'hybrid[ -]?dominant': 'hybrid',
        r'indica[ -]?dominant': 'indica',
        r'sativa[ -]?dominant': 'sativa'
    }
    
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    
    # Remove special characters except % for percentages
    text = re.sub(r"[^\w\s%]", " ", text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    return text

def init_index_service():
    global index, product_ids, products_data
    logger.info("Initializing index service...")
    
    products_data = list(get_products_collection().find(
        {}, {"_id": 1, "description": 1}
    ).max_time_ms(5000))
    
    if not products_data:
        raise ValueError("No products found in the database")
    
    model = get_embedding_model()
    
    # Preprocess descriptions before embedding
    descriptions = [preprocess_description(p["description"]) for p in products_data]
    embeddings = model.encode(descriptions, convert_to_tensor=False)
    
    dim = embeddings.shape[1]
    index = hnswlib.Index(space="cosine", dim=dim)
    
    if os.path.exists(Config.INDEX_FILE):
        try:
            index.load_index(Config.INDEX_FILE)
            index.set_ef(50)
            
            # Validate loaded index matches current data
            if index.get_current_count() != len(products_data):
                logger.warning("Index count mismatch - rebuilding...")
                create_new_index(embeddings)
        except Exception as e:
            logger.error(f"Index load failed: {e}")
            create_new_index(embeddings)
    else:
        create_new_index(embeddings)
    
    product_ids = [p["_id"] for p in products_data]
    logger.info(f"Index initialized with {len(product_ids)} products")

def create_new_index(embeddings):
    global index
    index.init_index(
        max_elements=len(products_data) * 2,  # Allow for growth
        ef_construction=200,
        M=16
    )
    index.add_items(embeddings)
    index.save_index(Config.INDEX_FILE)
    logger.info(f"Created new index with {len(products_data)} items")

def search_index(query_embedding, k):
    try:
        labels, distances = index.knn_query(query_embedding, k=k)
        return labels, distances
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise

def get_product_by_index(idx):
    if idx >= len(product_ids):
        raise IndexError("Index out of range")
    product_id = product_ids[idx]
    return next(p for p in products_data if p["_id"] == product_id)

def refresh_index():
    logger.info("Refreshing index...")
    init_index_service()
    return len(products_data)