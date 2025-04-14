from config import Config
import requests
from utils.logger import logger
from utils.ollama_monitor import OllamaMonitor

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


def ask_question(context, question):
    try:
        # Create and start the Ollama monitor
        monitor = OllamaMonitor().start_monitoring()
        
        # Use Ollama for responses
        prompt = f"""
        Based on the following context, please answer the question:
        
        Context: {context}
        
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
            
            # Return both the answer and the usage statistics
            return {
                "answer": answer,
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
            "usage_stats": {
                "error": str(e)
            }
        }
