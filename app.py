from flask import Flask, request, jsonify
import joblib
import numpy as np
import tldextract
import re
from urllib.parse import urlparse

app = Flask(__name__)

# Load the trained model
model = joblib.load('phishing_url_model.pkl')

@app.route('/')
def home():
    return "Phishing URL Detection API is up and running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the URL from the request
    url = request.json['url']
    
    # Preprocess the URL to extract features
    features = preprocess_url(url)  # Feature extraction from the URL
    prediction = model.predict([features])  # Make the prediction
    
    result = {
        'url': url,
        'prediction': 'phishing' if prediction == 1 else 'legitimate'
    }
    
    return jsonify(result)

def preprocess_url(url):
    """
    This function takes a URL and extracts the features that were used
    for training the model.
    """
    # Extract features from the URL
    features = []
    
    # 1. URL Length
    features.append(len(url))
    
    # 2. Number of '.' (dots) in the URL
    features.append(url.count('.'))
    
    # 3. Number of '-' (hyphens) in the URL
    features.append(url.count('-'))
    
    # 4. Check if the URL contains 'http' or 'https'
    features.append(1 if 'http' in url else 0)
    
    # 5. Check if the URL contains 'www'
    features.append(1 if 'www' in url else 0)
    
    # 6. Extract domain using tldextract
    ext = tldextract.extract(url)
    domain = ext.domain
    suffix = ext.suffix
    
    # 7. Length of the domain part (without the suffix)
    features.append(len(domain))
    
    # 8. Length of the suffix
    features.append(len(suffix))
    
    # 9. URL contains an IP address (a common phishing indicator)
    features.append(1 if re.match(r'^(https?://)?(\d{1,3}\.){3}\d{1,3}', url) else 0)
    
    # 10. Extract path from the URL
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # 11. Length of the path part of the URL
    features.append(len(path))
    
    # 12. Check if the URL has a suspicious keyword in the path
    suspicious_keywords = ['login', 'signin', 'account', 'update', 'secure']
    features.append(1 if any(keyword in path for keyword in suspicious_keywords) else 0)
    
    # 13. Count of query parameters in the URL (if any)
    features.append(url.count('?'))
    
    # 14. Count of fragments in the URL (if any)
    features.append(url.count('#'))
    
    # Convert the features list into a numpy array
    return np.array(features)

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment or default to 5000
    app.run(host='0.0.0.0', port=port)

