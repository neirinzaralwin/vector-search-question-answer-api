from flask import Blueprint, jsonify
from services.index_service import refresh_index
from utils.logger import logger
from datetime import datetime

bp = Blueprint('index', __name__, url_prefix='/')

@bp.route('/refresh-index', methods=['POST'])
def handle_refresh_index():
    logger.info("Received index refresh request")
    try:
        product_count = refresh_index()
        return jsonify({
            "status": "success",
            "message": "Index refreshed",
            "product_count": product_count,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Index refresh failed: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500