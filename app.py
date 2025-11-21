import requests
import os
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load environment variables from a .env file (if present)
load_dotenv() 

app = Flask(__name__)

# --- Configuration ---
# NOTE: In a real-world deployment, this key should be loaded securely.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

def preprocess_question(question: str) -> str:
    """
    Applies basic preprocessing steps to the user's question for display.
    """
    # 1. Lowercasing
    processed = question.lower().strip()
    # 2. Punctuation removal (keeping spaces)
    processed = re.sub(r'[^\w\s]', '', processed)
    
    tokens = processed.split()
    return " ".join(tokens)

def generate_answer(processed_query: str) -> dict:
    """
    Sends the processed query to the Gemini API and returns structured results.
    """
    if not GEMINI_API_KEY:
        return {
            "error": True,
            "answer": "GEMINI_API_KEY not configured on the server. Please check your .env file or environment variables.",
            "raw_response": "N/A"
        }

    system_instruction = "You are a friendly, web-based Question-Answering assistant. Provide a clear and helpful answer."
    
    payload = {
        "contents": [{"parts": [{"text": processed_query}]}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "tools": [{"google_search": {}}],
    }

    headers = {
        'Content-Type': 'application/json'
    }

    try:
        # Use exponential backoff for retries (simplified here, but important)
        response = requests.post(f"{API_URL}?key={GEMINI_API_KEY}", json=payload, headers=headers, timeout=60)
        
        # Check for non-200 status codes
        if response.status_code != 200:
            return {
                "error": True,
                "answer": f"API Error: Status {response.status_code}. Response: {response.text}",
                "raw_response": response.text
            }
        
        result = response.json()
        
        # Extract the text content
        answer_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', 'No answer generated.')
        
        return {
            "error": False,
            "answer": answer_text,
            "raw_response": response.text # Keep the raw response for the debug panel
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": True,
            "answer": f"Connection Error: Could not reach the API endpoint. {e}",
            "raw_response": str(e)
        }
    except Exception as e:
        return {
            "error": True,
            "answer": f"An unexpected server error occurred: {e}",
            "raw_response": str(e)
        }

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def handle_answer():
    """Handles the user question submitted via the web form."""
    data = request.get_json()
    user_question = data.get('question', '').strip()

    if not user_question:
        return jsonify({
            "error": True, 
            "message": "Please enter a question.", 
            "processed": "", 
            "answer": "No question provided.",
            "raw_response": ""
        }), 400

    # Part B: Preprocessing
    processed_query = preprocess_question(user_question)
    
    # Part B: LLM Query and Answer
    response_data = generate_answer(processed_query)
    
    # Use the status code from the API response handler
    status_code = 200 if not response_data["error"] else 500
    message = response_data["answer"]

    # Return the full result structure to the frontend
    return jsonify({
        "error": response_data["error"],
        "message": message,
        "processed": processed_query,
        "raw_response": response_data["raw_response"]
    }), status_code

if __name__ == '__main__':
    if not GEMINI_API_KEY:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!! WARNING: GEMINI_API_KEY is NOT set. The API calls will fail. !!")
        print("!! Please ensure you have a .env file with GEMINI_API_KEY='your_key' !!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
    # Use 0.0.0.0 for hosting/deployment compatibility
    app.run(debug=True, host='0.0.0.0', port=5000)