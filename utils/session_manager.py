from datetime import datetime, timedelta
from utils.logger import logger

# Dictionary to store user sessions 
# Format: {session_id: {"last_updated": datetime, "product_context": {...}, "conversation_history": [...]}}
active_sessions = {}

# Session timeout in minutes
SESSION_TIMEOUT = 30

def get_session(session_id):
    """Get session data for a specific session ID, or create new if doesn't exist"""
    cleanup_expired_sessions()
    
    if session_id not in active_sessions:
        logger.info(f"Creating new session: {session_id}")
        active_sessions[session_id] = {
            "last_updated": datetime.now(),
            "product_context": None,
            "conversation_history": []
        }
    else:
        # Update the last_updated time
        active_sessions[session_id]["last_updated"] = datetime.now()
    
    return active_sessions[session_id]

def update_session_context(session_id, product_context, question, answer):
    """Update session with new product context and conversation history"""
    if session_id not in active_sessions:
        get_session(session_id)
    
    active_sessions[session_id]["product_context"] = product_context
    
    # Add the QA pair to conversation history (keep last 5 for context)
    active_sessions[session_id]["conversation_history"].append({
        "question": question,
        "answer": answer,
        "timestamp": datetime.now().isoformat()
    })
    
    # Limit conversation history to last 5 exchanges
    if len(active_sessions[session_id]["conversation_history"]) > 5:
        active_sessions[session_id]["conversation_history"] = active_sessions[session_id]["conversation_history"][-5:]

def cleanup_expired_sessions():
    """Remove sessions that have been inactive for longer than SESSION_TIMEOUT"""
    now = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in active_sessions.items():
        if now - session_data["last_updated"] > timedelta(minutes=SESSION_TIMEOUT):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        logger.info(f"Removing expired session: {session_id}")
        del active_sessions[session_id]