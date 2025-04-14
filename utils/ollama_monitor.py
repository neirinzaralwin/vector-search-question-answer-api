import subprocess
import json
import psutil
import requests
from config import Config
from utils.logger import logger


def get_ollama_process():
    """Find the Ollama process in the system processes"""
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.info['name'] == 'ollama' or (process.info['cmdline'] and 'ollama' in ' '.join(process.info['cmdline'])):
            return process.pid
    return None


def get_model_info():
    """Get information about the currently loaded Ollama models"""
    try:
        response = requests.get(f"{Config.OLLAMA_URL}/api/tags")
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Failed to get model info: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return None


def get_ollama_memory_usage():
    """Get the memory usage of the Ollama process and the loaded models"""
    try:
        pid = get_ollama_process()
        if not pid:
            logger.warning("Ollama process not found")
            return {
                "status": "error", 
                "message": "Ollama process not found. Make sure Ollama is running."
            }
            
        # Get process memory info
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        # Get model info
        model_info = get_model_info()
        
        return {
            "status": "success",
            "process": {
                "pid": pid,
                "rss_memory_mb": memory_info.rss / (1024 * 1024),  # Convert bytes to MB
                "vms_memory_mb": memory_info.vms / (1024 * 1024),  # Convert bytes to MB
                "memory_percent": memory_percent
            },
            "models": model_info["models"] if model_info else []
        }
    except Exception as e:
        logger.error(f"Error getting Ollama memory usage: {str(e)}")
        return {"status": "error", "message": str(e)}


def print_memory_usage():
    """Print the memory usage in a readable format"""
    usage = get_ollama_memory_usage()
    
    if usage["status"] == "error":
        print(f"Error: {usage['message']}")
        return
    
    print("\n===== OLLAMA MEMORY USAGE =====")
    print(f"PID: {usage['process']['pid']}")
    print(f"RSS Memory: {usage['process']['rss_memory_mb']:.2f} MB")
    print(f"Virtual Memory: {usage['process']['vms_memory_mb']:.2f} MB")
    print(f"Memory Percent: {usage['process']['memory_percent']:.2f}%")
    
    print("\n===== LOADED MODELS =====")
    for model in usage["models"]:
        size_mb = model.get("size", 0) / (1024 * 1024)  # Convert to MB
        print(f"Model: {model['name']}, Size: {size_mb:.2f} MB, Modified: {model.get('modified', 'unknown')}")
    
    print("\nNote: RSS (Resident Set Size) is the actual physical memory used by the process")


if __name__ == "__main__":
    print_memory_usage()