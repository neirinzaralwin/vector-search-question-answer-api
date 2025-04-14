from config import Config
import requests
from utils.logger import logger
from utils.ollama_monitor import OllamaMonitor
from services import index_service, embedding_service
import re

# Constants
CONTEXT_SIMILARITY_THRESHOLD = 0.75  # Threshold to determine if we need to change product context
PRODUCT_NAME_PATTERN = re.compile(r'(strain|kush|og|haze|purple|blue|green|cake|dream|cookies|diesel|widow|skunk|cheese|lemon|herer|jack|northern|light|purple|sour|sweet|berry|fruit)', re.IGNORECASE)

# List of common strain names to improve search accuracy
COMMON_STRAIN_NAMES = [
    "birthday cake", "wedding cake", "purple haze", "blue dream", "sour diesel", 
    "girl scout cookies", "northern lights", "og kush", "pineapple express",
    "white widow", "ak-47", "green crack", "granddaddy purple", "jack herer",
    "bubba kush", "durban poison", "gelato", "gorilla glue", "super lemon haze"
]

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
            
            if is_new_topic and contains_potential_product_reference(question):
                # Only search for a new product if the question might contain a product reference
                logger.info(f"Potential topic change detected in query: '{question}'")
                context_used, new_product_detected = get_product_context_from_query(question)
            
            # If we didn't find a new product context, use the existing one
            if not new_product_detected:
                context_used = session_data["product_context"]
                logger.info(f"Using existing product context: {get_product_name(context_used)}")
        
        # Special handling for queries that clearly ask about specific strains
        detected_strain_name = extract_strain_name(question)
        if detected_strain_name and (context_used is None or get_product_name(context_used).lower() != detected_strain_name.lower()):
            # Try again with the explicit strain name with higher threshold
            logger.info(f"Explicit strain '{detected_strain_name}' detected in query, searching more specifically")
            direct_context, direct_found = get_product_by_strain_name(detected_strain_name)
            if direct_found:
                context_used = direct_context
                new_product_detected = True
        
        if context_used is None:
            # No context found, use a generic response
            context = f"You are a helpful customer support agent for a cannabis dispensary. The customer asked about {extract_strain_name(question) or 'a product'}, but we couldn't find specific information about it."
            logger.warning(f"No product context found for query: '{question}'")
        else:
            # Use the product description as context
            context = context_used["description"]
        
        # Build conversation history context if available
        conversation_context = ""
        product_name = get_product_name(context_used) if context_used else (extract_strain_name(question) or "this product")
        
        if session_data and session_data["conversation_history"]:
            conversation_history = session_data["conversation_history"]
            conversation_context = "\nPrevious conversation:\n"
            for exchange in conversation_history:
                conversation_context += f"Customer: {exchange['question']}\nSupport: {exchange['answer']}\n"
        
        # Use Ollama for responses with ultra-concise customer support style
        prompt = f"""
        You work at a cannabis dispensary answering customer questions. Keep responses between 10-50 words unless absolutely necessary.

        Product facts: {context}
        
        {conversation_context}
        
        Customer: {question}
        
        Rules for your response:
        1. Be direct and concise (10-50 words)
        2. No greetings, no sign-offs
        3. No "I'd be happy to help" or similar phrases
        4. No unnecessary politeness
        5. Just answer the specific question about {product_name}
        6. If you don't have info, just say so briefly
        7. Skip educational explanations unless specifically asked
        
        Reply:
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
                    "max_tokens": 150  # Further reduced for shorter responses
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
            "answer": f"Sorry, couldn't process your question. Try again.",
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
        
        # Search the index with top-3 results for better matching
        labels, distances = index_service.search_index(query_embedding, k=3)
        
        # Check if we have a good match - lower threshold for better recall
        if len(labels[0]) > 0 and distances[0][0] < 0.5:  # Lower distance means more similar
            product = index_service.get_product_by_index(labels[0][0])
            logger.info(f"Found product context for query (similarity score: {1-distances[0][0]:.2f})")
            return product, True
        else:
            logger.info(f"No relevant product found for query: '{query}'")
            return None, False
    except Exception as e:
        logger.error(f"Error searching for product context: {str(e)}")
        return None, False

def get_product_by_strain_name(strain_name):
    """Directly search for a product by strain name"""
    try:
        # Try to find exact matches in the product data
        all_products = index_service.products_data
        
        # Normalize the strain name for comparison
        normalized_name = strain_name.lower().strip()
        
        # First look for exact matches at the beginning of descriptions
        for product in all_products:
            desc = product.get("description", "").lower()
            if desc.startswith(normalized_name) or f"{normalized_name} strain" in desc:
                logger.info(f"Found exact match for strain name: {strain_name}")
                return product, True
        
        # If no exact match, try word-by-word matching
        for product in all_products:
            desc = product.get("description", "").lower()
            
            # Check if all words in the strain name appear in the first 100 chars of description
            words = normalized_name.split()
            if all(word in desc[:100] for word in words):
                logger.info(f"Found partial match for strain name: {strain_name}")
                return product, True
                
        # If still no match, try embedding search with the strain name specifically
        query = f"{strain_name} strain"
        query_embedding = embedding_service.encode_query(query)
        labels, distances = index_service.search_index(query_embedding, k=1)
        
        if len(labels[0]) > 0 and distances[0][0] < 0.4:  # Stricter threshold for name matching
            product = index_service.get_product_by_index(labels[0][0])
            logger.info(f"Found strain via embedding: {strain_name} (score: {1-distances[0][0]:.2f})")
            return product, True
            
        return None, False
    except Exception as e:
        logger.error(f"Error in strain name search: {str(e)}")
        return None, False

def detect_topic_change(question, current_product_context):
    """
    Determine if the user's question is about a different product than the current context.
    
    Returns True if the question likely refers to a new product.
    """
    if current_product_context is None:
        return True
        
    try:
        # If the query contains terms that clearly indicate it's a follow-up question, maintain context
        follow_up_indicators = ["it", "this strain", "this product", "effects", "taste", "smell", "thc", "cbd", "potency"]
        
        # Check for follow-up indicators
        for indicator in follow_up_indicators:
            if indicator.lower() in question.lower():
                logger.info(f"Detected follow-up indicator '{indicator}' in question")
                return False
        
        # Get product name from context
        product_name = get_product_name(current_product_context)
        if product_name and product_name.lower() in question.lower():
            logger.info(f"Product name '{product_name}' found in question, maintaining context")
            return False
            
        # Encode the question and current product description
        question_embedding = embedding_service.encode_query(question)
        description_embedding = embedding_service.encode_query(current_product_context["description"])
        
        # Calculate cosine similarity between question and current product
        similarity_score = index_service.index.space.get_distance(question_embedding, description_embedding)
        similarity_score = 1 - similarity_score  # Convert distance to similarity
        
        # Check if the question is sufficiently dissimilar from current context
        logger.info(f"Question similarity to current product: {similarity_score:.2f}")
        
        is_new_topic = similarity_score < CONTEXT_SIMILARITY_THRESHOLD
        
        # Additional check: if there's no potential product reference, keep the current context
        if is_new_topic and not contains_potential_product_reference(question):
            logger.info("No product reference found in query, maintaining current context")
            is_new_topic = False
        
        if is_new_topic:
            logger.info("Detected potential topic change")
        
        return is_new_topic
    except Exception as e:
        logger.error(f"Error detecting topic change: {str(e)}")
        return False  # Default to keeping current context on error

def contains_potential_product_reference(text):
    """Check if the text potentially contains a reference to a cannabis product"""
    return bool(PRODUCT_NAME_PATTERN.search(text))

def get_product_name(product_context):
    """Extract a product name from product context"""
    if not product_context:
        return None
    
    # Try to extract name from description
    try:
        # Look for product name at the beginning of description
        description = product_context.get("description", "")
        first_sentence = description.split('.')[0]
        
        # Look for common patterns like "X strain" or "X is a" at the beginning
        if ", also known as" in first_sentence:
            name = first_sentence.split(', also known as')[0].strip()
        elif " is a " in first_sentence:
            name = first_sentence.split(' is a ')[0].strip()
        elif " strain" in first_sentence.lower():
            name = first_sentence.split(' strain')[0].strip()
        else:
            # Fall back to first few words (likely the strain name)
            words = first_sentence.split()
            name = ' '.join(words[:2]) if len(words) > 1 else words[0]
            
        return name
    except:
        # If something goes wrong, return generic name
        return "this cannabis product"

def extract_strain_name(text):
    """Extract potential strain name from user query"""
    # Check for common patterns like "about X strain" or "tell me about X"
    text = text.lower()
    
    # Look for strain names from our known list
    for strain in COMMON_STRAIN_NAMES:
        if strain in text:
            return strain
    
    # Look for patterns like "about X strain" or "about the X strain"
    about_pattern = re.compile(r'about\s+(?:the\s+)?([a-z\s]+?)(?:\s+strain)?[.?]?$', re.IGNORECASE)
    match = about_pattern.search(text)
    if match:
        return match.group(1).strip()
    
    # Look for patterns like "what is X strain" or "what's X"
    what_pattern = re.compile(r'what(?:\s+is|\s*\'s)\s+(?:the\s+)?([a-z\s]+?)(?:\s+strain)?[.?]?$', re.IGNORECASE)
    match = what_pattern.search(text)
    if match:
        return match.group(1).strip()
    
    # Extract text between quotes if present (e.g., "Birthday Cake")
    quote_pattern = re.compile(r'"([^"]+)"', re.IGNORECASE)
    match = quote_pattern.search(text)
    if match:
        return match.group(1).strip()
    
    return None
