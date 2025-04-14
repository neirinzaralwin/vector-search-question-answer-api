from flask import Blueprint, request, jsonify
from utils.logger import logger
from services import qa_service
from utils.ollama_monitor import get_ollama_memory_usage

bp = Blueprint('qa', __name__, url_prefix='/')

@bp.route('/qa', methods=['POST'])
def ask_question():
    
    question = request.args.get('q', '').strip()

    if not question:
        logger.warning("Empty question received")
        return jsonify({"error": "Missing parameter", "message": "Question parameter 'q' is required"}), 400
    
    context = "Kandy Kush, also known as \"Candy Kush,\" is a hybrid marijuana strain and a favorite of DNA Genetics' Reserva Privada line that combines two California classics, OG Kush (thought to be the \"Christopher Wallace\" cut) and Trainwreck, to make a tasty indica-dominant hybrid (although sativa phenotypes displaying more of the Trainwreck structure have been noted). Like the name suggests, the flavor is sweet like candy with a strong lemon-Kush scent. Popular with medicinal growers, Kandy Kush provides a potent body high with pronounced pain relief.\nType: Hybrid\nTHC: 18%\nCBD: 0%"

    try:
        # Get answer from Ollama
        answer = qa_service.ask_question(context, question)
        
        # Get Ollama memory usage
        ollama_usage = get_ollama_memory_usage()
        
        # Prepare response with both answer and memory usage statistics
        response_data = {
            "question": question,
            "answer": answer,
            "ollama_usage": {
                "status": ollama_usage["status"]
            }
        }
        
        # Add memory details if available
        if ollama_usage["status"] == "success":
            response_data["ollama_usage"].update({
                "memory_mb": round(ollama_usage["process"]["rss_memory_mb"], 2),
                "memory_percent": round(ollama_usage["process"]["memory_percent"], 2),
                "model": {
                    "name": ollama_usage["models"][0]["name"] if ollama_usage["models"] else "unknown"
                }
            })
        else:
            response_data["ollama_usage"]["message"] = ollama_usage.get("message", "Unknown error")
            
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        return jsonify({"error": "QA failed", "message": str(e)}), 500

