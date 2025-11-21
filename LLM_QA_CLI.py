import os
import string
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def preprocess_text(text):
    """
    Applies basic preprocessing:
    1. Lowercasing
    2. Removing punctuation
    """
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def query_llm(prompt):
    """
    Sends the prompt to the Gemini API and returns the response.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY environment variable not found. Please set it in a .env file."

    try:
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash as it is a current and fast model
        model = genai.GenerativeModel('models/gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with LLM API: {str(e)}"

def main():
    print("--- LLM Q&A CLI Application ---")
    print("Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting...")
            break
        
        if not user_input.strip():
            continue

        # Preprocess
        processed_input = preprocess_text(user_input)
        print(f"[Processed Input]: {processed_input}")

        # Query LLM
        print("Querying LLM...")
        
        # We send the processed input as per requirements
        final_answer = query_llm(processed_input)
        
        print("\n--- LLM Answer ---")
        print(final_answer)
        print("------------------")

if __name__ == "__main__":
    main()
