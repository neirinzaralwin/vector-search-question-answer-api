from flask import Blueprint, request, jsonify
from utils.logger import logger
from services import qa_service
from config import Config

bp = Blueprint('qa', __name__, url_prefix='/')

@bp.route('/qa', methods=['POST'])
def ask_question():
    
    question = request.args.get('q', '').strip()

    if not question:
        logger.warning("Empty question received")
        return jsonify({"error": "Missing parameter", "message": "Question parameter 'q' is required"}), 400
    
    context = "Kandy Kush, also known as \"Candy Kush,\" is a hybrid marijuana strain and a favorite of DNA Genetics' Reserva Privada line that combines two California classics, OG Kush (thought to be the \"Christopher Wallace\" cut) and Trainwreck, to make a tasty indica-dominant hybrid (although sativa phenotypes displaying more of the Trainwreck structure have been noted). Like the name suggests, the flavor is sweet like candy with a strong lemon-Kush scent. Popular with medicinal growers, Kandy Kush provides a potent body high with pronounced pain relief.\nType: Hybrid\nTHC: 18%\nCBD: 0%"

    try:
        # Get answer from Ollama with usage statistics
        response = qa_service.ask_question(context, question)
        
        # Prepare response with answer, peak RAM usage, and response time
        response_data = {
            "question": question,
            "answer": response["answer"],
            "model_stats": {
                "model": Config.OLLAMA_MODEL,
                "peak_memory_mb": response["usage_stats"]["peak_memory_mb"],
                "peak_memory_percent": response["usage_stats"]["peak_memory_percent"],
                "response_time_seconds": response["usage_stats"]["duration_seconds"]
            }
        }
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        return jsonify({"error": "QA failed", "message": str(e)}), 500

