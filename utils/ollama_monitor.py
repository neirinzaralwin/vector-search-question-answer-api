import psutil
import requests
import time
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


class OllamaMonitor:
    """Monitor Ollama resource usage during generation"""
    
    def __init__(self):
        self.peak_memory_mb = 0
        self.peak_memory_percent = 0
        self.start_time = None
        self.end_time = None
        self.monitoring = False
        self.pid = get_ollama_process()
        self.process = None
        if self.pid:
            self.process = psutil.Process(self.pid)
        
    def start_monitoring(self):
        """Start monitoring Ollama resource usage"""
        self.start_time = time.time()
        self.peak_memory_mb = 0
        self.peak_memory_percent = 0
        self.monitoring = True
        # Get initial memory reading
        if self.process:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            self.peak_memory_mb = memory_info.rss / (1024 * 1024)
            self.peak_memory_percent = memory_percent
        return self
        
    def update_peak_memory(self):
        """Update peak memory if current usage is higher"""
        if not self.monitoring or not self.process:
            return
        
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            current_memory_mb = memory_info.rss / (1024 * 1024)
            
            if current_memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = current_memory_mb
                self.peak_memory_percent = memory_percent
        except:
            # Process might have exited
            pass
            
    def stop_monitoring(self):
        """Stop monitoring and return stats"""
        self.update_peak_memory()  # Final check for peak memory
        self.end_time = time.time()
        self.monitoring = False
        
        duration_seconds = self.end_time - self.start_time
        
        return {
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "peak_memory_percent": round(self.peak_memory_percent, 2),
            "duration_seconds": round(duration_seconds, 3)
        }