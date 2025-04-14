from flask import Blueprint, request, jsonify, session
from utils.logger import logger
from services import qa_service
from config import Config
from utils.session_manager import get_session, update_session_context
import uuid

bp = Blueprint('qa', __name__, url_prefix='/')

@bp.route('/qa', methods=['POST'])
def ask_question():
    question = request.args.get('q', '').strip()
    
    if not question:
        logger.warning("Empty question received")
        return jsonify({"error": "Missing parameter", "message": "Question parameter 'q' is required"}), 400

    # Get or create session ID
    session_id = request.args.get('session_id')
    if not session_id:
        session_id = str(uuid.uuid4())
        logger.info(f"No session ID provided, created new session: {session_id}")
    
    # Get session data for this user
    session_data = get_session(session_id)
    
    try:
        # Get answer using the session context and question
        qa_response = qa_service.ask_question(question, session_data)
        
        # Extract data
        answer = qa_response["answer"]
        context_used = qa_response["context_used"]
        new_product_detected = qa_response["new_product_detected"]
        usage_stats = qa_response["usage_stats"]
        
        # Update session with new context and conversation history
        update_session_context(session_id, context_used, question, answer)
        
        # Prepare response with answer, context info, and performance stats
        response_data = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "context_changed": new_product_detected,
            "model_stats": {
                "model": Config.OLLAMA_MODEL,
                "peak_memory_mb": usage_stats["peak_memory_mb"],
                "peak_memory_percent": usage_stats["peak_memory_percent"],
                "response_time_seconds": usage_stats["duration_seconds"]
            }
        }
        
        # Add product info if available
        if context_used:
            response_data["product"] = {
                "id": str(context_used["_id"]),
                "description": context_used["description"][:100] + "..." if len(context_used["description"]) > 100 else context_used["description"]
            }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        return jsonify({
            "error": "QA failed", 
            "message": str(e),
            "session_id": session_id
        }), 500

