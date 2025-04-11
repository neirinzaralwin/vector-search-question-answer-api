from flask import Blueprint, request, jsonify
from services import index_service, embedding_service
from utils.logger import logger

bp = Blueprint('search', __name__, url_prefix='/')

@bp.route('/search', methods=['GET'])
def search_products():
    logger.info(f"Search request received - Query: {request.args.get('q', '')}, K: {request.args.get('k', '3')}")
    
    query_text = request.args.get('q', '').strip()
    try:
        k = min(int(request.args.get('k', 3)), 20)
    except ValueError:
        logger.warning(f"Invalid k value provided: {request.args.get('k')}")
        return jsonify({"error": "Invalid parameter", "message": "Parameter 'k' must be an integer"}), 400
    
    if not query_text:
        logger.warning("Empty query received")
        return jsonify({"error": "Missing parameter", "message": "Query parameter 'q' is required"}), 400
    
    if len(query_text) > 500:
        logger.warning(f"Query too long ({len(query_text)} characters)")
        return jsonify({"error": "Invalid parameter", "message": "Query text is too long (max 500 characters)"}), 400
    
    try:
        query_embedding = embedding_service.encode_query(query_text)
        labels, distances = index_service.search_index(query_embedding, k)
        
        results = []
        for idx, dist in zip(labels[0], distances[0]):
            try:
                product = index_service.get_product_by_index(idx)
                results.append({
                    "id": str(product['_id']),
                    "description": product['description'],
                    "score": float(1 - dist)
                })
            except (IndexError, StopIteration) as e:
                logger.warning(f"Product lookup failed for index {idx}: {str(e)}")
                continue
        
        logger.info(f"Search completed with {len(results)} results")
        return jsonify({"query": query_text, "k": k, "results": results})
    
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({"error": "Search failed", "message": str(e)}), 500