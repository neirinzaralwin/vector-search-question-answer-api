from flask import Blueprint, jsonify
from services.database_service import get_products_collection
from services.index_service import index, products_data
from utils.logger import logger
from datetime import datetime

bp = Blueprint('health', __name__, url_prefix='/')

@bp.route('/health', methods=['GET'])
def health_check():
    try:
        # Database check
        get_products_collection().find_one()
        # Index check
        if not index or not products_data:
            raise RuntimeError("Index not initialized")
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500