from flask import Flask, render_template, request
import pickle
import os
import requests
import re
import urllib.parse
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

app = Flask(__name__)

# Load the trained model and vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    with open(VECTORIZER_PATH, 'rb') as file:
        vectorizer = pickle.load(file)
else:
    print("Warning: model.pkl or vectorizer.pkl not found. Please run train.py first.")

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

def extract_keywords(text):
    """Extracts main keywords from the input to search the news API."""
    stop_words = {'is', 'the', 'of', 'and', 'to', 'a', 'in', 'that', 'it', 'for', 'on', 'with', 'as', 'this', 'was', 'at', 'by', 'an', 'be', 'from'}
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    keywords = [w for w in words if w not in stop_words]
    # Return top 2 keywords to form a broader search query
    return " ".join(keywords[:2])

def search_news_api(query):
    """Searches NewsAPI for related articles, with graceful failure handling."""
    if not NEWS_API_KEY:
        return {"status": "error", "message": "API Key missing in .env file."}
    
    url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote(query)}&apiKey={NEWS_API_KEY}&language=en&sortBy=relevancy&pageSize=3"
    try:
        response = requests.get(url, timeout=5) # 5 second timeout to prevent hanging
        if response.status_code == 200:
            data = response.json()
            return {"status": "success", "articles": data.get("articles", [])}
        else:
            return {"status": "error", "message": "API request failed."}
    except Exception:
        # Handles internet unavailable or timeout
        return {"status": "error", "message": "Unable to fetch live resources currently."}

def get_latest_news():
    """Fetches the latest general headlines for the sidebar."""
    if not NEWS_API_KEY:
        return []
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}&pageSize=5"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json().get("articles", [])
    except Exception:
        pass
    return []

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    resource_status = None
    related_sources = []
    user_input = ""
    latest_news = get_latest_news()
    
    if request.method == 'POST':
        user_input = request.form.get('news_text', '')
        
        if user_input.strip() and model and vectorizer:
            # 1. Convert input to TF-IDF features
            transformed_input = vectorizer.transform([user_input])
            
            # 2. Predict fake (0) or real (1)
            pred = model.predict(transformed_input)[0]
            prediction = "REAL NEWS" if pred == 1 else "FAKE NEWS"
            
            # 3. Calculate Model Confidence
            proba = model.predict_proba(transformed_input)[0]
            confidence = round(max(proba) * 100, 2)
            
            # 4. Search Live News
            keywords = extract_keywords(user_input)
            news_results = search_news_api(keywords) if keywords else {"status": "error"}
            
            if news_results["status"] == "success":
                articles = news_results["articles"]
                if len(articles) > 0:
                    resource_status = "Resource Available"
                    related_sources = articles
                else:
                    resource_status = "No Resource Available"
            else:
                # Handle API failure
                resource_status = news_results.get("message", "Unable to fetch live resources currently.")
                
    return render_template('index.html', 
                           prediction=prediction, 
                           confidence=confidence, 
                           resource_status=resource_status, 
                           related_sources=related_sources, 
                           user_input=user_input,
                           latest_news=latest_news)

if __name__ == '__main__':
    app.run(debug=True)
