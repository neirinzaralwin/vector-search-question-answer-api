from config import Config
import requests
from utils.logger import logger
from utils.ollama_monitor import OllamaMonitor
from services import index_service, embedding_service

# Constants
CONTEXT_SIMILARITY_THRESHOLD = 0.75  # Threshold to determine if we need to change product context

def init_qa_service():
    try:
        # Check Ollama connectivity
        try:
            response = requests.get(f"{Config.OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                logger.info(f"Ollama service detected and ready with model: {Config.OLLAMA_MODEL}")
            else:
                logger.error(f"Ollama service responded with status code: {response.status_code}")
                raise RuntimeError(f"Ollama service unavailable. Status code: {response.status_code}")
        except requests.exceptions.ConnectionError:
            logger.error("Couldn't connect to Ollama service")
            raise RuntimeError(f"Could not connect to Ollama at {Config.OLLAMA_URL}. Please make sure Ollama is running.")
    except Exception as e:
        logger.error(f"Failed to initialize QA service: {str(e)}")
        raise
    return None  # No pipeline needed for Ollama

def ask_question(question, session_data=None):
    """
    Answer a question using the context from the current session or by
    searching for a relevant product if no session exists or a topic change is detected.
    
    Args:
        question: The user's question
        session_data: Dictionary containing product_context and conversation_history
    
    Returns:
        Dictionary with answer, context used (product info), and usage statistics
    """
    try:
        # Create and start the Ollama monitor
        monitor = OllamaMonitor().start_monitoring()
        
        # Track if we found a new product context
        new_product_detected = False
        context_used = None
        
        # Check if we need to search for a new product context
        if session_data is None or session_data["product_context"] is None:
            # No existing context - search for product by query
            logger.info(f"No existing context, searching for product based on query: '{question}'")
            context_used, new_product_detected = get_product_context_from_query(question)
        else:
            # Check if user is asking about a new product or using the existing context
            is_new_topic = detect_topic_change(question, session_data["product_context"])
            
            if is_new_topic:
                # Search for a new product context
                logger.info(f"Potential topic change detected in query: '{question}'")
                context_used, new_product_detected = get_product_context_from_query(question)
            
            # If we didn't find a new product context, use the existing one
            if not new_product_detected:
                context_used = session_data["product_context"]
                logger.info("Using existing product context")
        
        if context_used is None:
            # No context found, use a generic response
            context = "You are an AI assistant that answers questions about cannabis products."
            logger.warning(f"No product context found for query: '{question}'")
        else:
            # Use the product description as context
            context = context_used["description"]
        
        # Build conversation history context if available
        conversation_context = ""
        if session_data and session_data["conversation_history"]:
            conversation_history = session_data["conversation_history"]
            conversation_context = "\nPrevious conversation:\n"
            for exchange in conversation_history:
                conversation_context += f"Q: {exchange['question']}\nA: {exchange['answer']}\n"
        
        # Use Ollama for responses with enhanced prompt
        prompt = f"""
        Based on the following context about a cannabis product, please answer the question:
        
        Product Context: {context}
        
        {conversation_context}
        
        Question: {question}
        
        Answer:
        """
        
        # Periodically check and update peak memory usage during request
        monitor.update_peak_memory()
        
        response = requests.post(
            f"{Config.OLLAMA_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 300
                }
            }
        )
        
        # Check memory usage after response
        monitor.update_peak_memory()
        
        if response.status_code == 200:
            result = response.json()
            answer = result["response"].strip()
            
            # Stop monitoring and get usage stats
            usage_stats = monitor.stop_monitoring()
            
            # Return the answer, context used, and the usage statistics
            return {
                "answer": answer,
                "context_used": context_used,
                "new_product_detected": new_product_detected,
                "usage_stats": usage_stats
            }
        else:
            # Stop monitoring even if there was an error
            monitor.stop_monitoring()
            
            logger.error(f"Ollama request failed: {response.status_code} - {response.text}")
            raise RuntimeError(f"Failed to get response from Ollama: {response.text}")
    except Exception as e:
        logger.error(f"Error during question answering: {str(e)}")
        return {
            "answer": f"I'm sorry, I couldn't process your question: {str(e)}",
            "context_used": None,
            "new_product_detected": False,
            "usage_stats": {
                "error": str(e)
            }
        }

def get_product_context_from_query(query):
    """Search for a product based on the query and return its context"""
    try:
        # Encode the query for vector search
        query_embedding = embedding_service.encode_query(query)
        
        # Search the index with top-1 result
        labels, distances = index_service.search_index(query_embedding, k=1)
        
        # Check if we have a good match
        if len(labels[0]) > 0 and distances[0][0] < 0.4:  # Lower distance means more similar
            product = index_service.get_product_by_index(labels[0][0])
            logger.info(f"Found product context for query (similarity score: {1-distances[0][0]:.2f})")
            return product, True
        else:
            logger.info(f"No relevant product found for query: '{query}'")
            return None, False
    except Exception as e:
        logger.error(f"Error searching for product context: {str(e)}")
        return None, False

def detect_topic_change(question, current_product_context):
    """
    Determine if the user's question is about a different product than the current context.
    
    Returns True if the question likely refers to a new product.
    """
    if current_product_context is None:
        return True
        
    try:
        # Encode the question and current product description
        question_embedding = embedding_service.encode_query(question)
        description_embedding = embedding_service.encode_query(current_product_context["description"])
        
        # Calculate cosine similarity between question and current product
        similarity_score = index_service.index.space.get_distance(question_embedding, description_embedding)
        similarity_score = 1 - similarity_score  # Convert distance to similarity
        
        # Check if the question is sufficiently dissimilar from current context
        logger.info(f"Question similarity to current product: {similarity_score:.2f}")
        
        is_new_topic = similarity_score < CONTEXT_SIMILARITY_THRESHOLD
        if is_new_topic:
            logger.info("Detected potential topic change")
        
        return is_new_topic
    except Exception as e:
        logger.error(f"Error detecting topic change: {str(e)}")
        return False  # Default to keeping current context on error
