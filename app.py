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
    
    # 2. Length of hostname
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname if parsed_url.hostname else ''
    features.append(len(hostname))
    
    # 3. Check if the URL contains an IP address
    features.append(1 if re.match(r'^(https?://)?(\d{1,3}\.){3}\d{1,3}', url) else 0)
    
    # 4. Number of dots in the URL
    features.append(url.count('.'))
    
    # 5. Number of hyphens in the URL
    features.append(url.count('-'))
    
    # 6. Number of '@' symbols in the URL
    features.append(url.count('@'))
    
    # 7. Number of '?' symbols (query parameters)
    features.append(url.count('?'))
    
    # 8. Number of '&' symbols in the URL (logical 'AND' operations)
    features.append(url.count('&'))
    
    # 9. Number of '=' symbols (logical 'OR' operations)
    features.append(url.count('='))
    
    # 10. Number of underscores in the URL
    features.append(url.count('_'))
    
    # 11. Number of tilde (~) symbols in the URL
    features.append(url.count('~'))
    
    # 12. Number of percentage (%) symbols in the URL
    features.append(url.count('%'))
    
    # 13. Number of slashes (/) in the URL
    features.append(url.count('/'))
    
    # 14. Number of stars (*) in the URL
    features.append(url.count('*'))
    
    # 15. Number of colons (:) in the URL
    features.append(url.count(':'))
    
    # 16. Number of commas (,) in the URL
    features.append(url.count(','))
    
    # 17. Number of semicolons (;) in the URL
    features.append(url.count(';'))
    
    # 18. Number of dollar ($) symbols in the URL
    features.append(url.count('$'))
    
    # 19. Number of spaces in the URL
    features.append(url.count(' '))
    
    # 20. Check if the URL contains 'www'
    features.append(1 if 'www' in url else 0)
    
    # 21. Check if the URL contains '.com'
    features.append(1 if '.com' in url else 0)
    
    # 22. Check if the URL has double slashes (//)
    features.append(1 if '//' in url else 0)
    
    # 23. Check if the path of the URL contains 'http'
    features.append(1 if 'http' in parsed_url.path else 0)
    
    # 24. Check if 'https' is part of the URL (secure connection)
    features.append(1 if 'https' in url else 0)
    
    # 25. Ratio of digits in the URL
    features.append(sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0)
    
    # 26. Ratio of digits in the hostname
    features.append(sum(c.isdigit() for c in hostname) / len(hostname) if len(hostname) > 0 else 0)
    
    # 27. Check for Punycode (indicates non-ASCII characters)
    features.append(1 if re.match(r'xn--', url) else 0)
    
    # 28. Check for port in the URL
    features.append(1 if parsed_url.port else 0)
    
    # 29. Check if the TLD is present in the path
    features.append(1 if parsed_url.path and parsed_url.path.endswith(tldextract.extract(url).suffix) else 0)
    
    # 30. Check if the TLD is present in the subdomain
    features.append(1 if parsed_url.hostname and tldextract.extract(url).subdomain else 0)
    
    # 31. Abnormal subdomain structure (e.g., containing subdomains like 'secure', 'login')
    features.append(1 if any(keyword in parsed_url.hostname for keyword in ['secure', 'login', 'signin']) else 0)
    
    # 32. Number of subdomains in the hostname
    features.append(parsed_url.hostname.count('.') - 1 if parsed_url.hostname else 0)
    
    # 33. Check for prefix/suffix (e.g., 'www')
    features.append(1 if parsed_url.hostname and parsed_url.hostname.startswith('www') else 0)
    
    # 34. Random domain name (using randomness or long combinations)
    features.append(1 if re.search(r'([a-z0-9]{8,})', parsed_url.hostname) else 0)
    
    # 35. URL shortened (using URL shortening services)
    features.append(1 if any(service in url for service in ['bit.ly', 'goo.gl']) else 0)
    
    # 36. Path contains an extension (e.g., .html, .php, etc.)
    features.append(1 if '.' in parsed_url.path else 0)
    
    # 37. Number of redirections in the URL
    features.append(url.count('redir'))
    
    # 38. Number of external redirections in the URL
    features.append(url.count('http') if parsed_url.hostname != hostname else 0)
    
    # 39. Length of raw words in the URL
    features.append(len([word for word in url.split('/') if len(word) > 0]))
    
    # 40. Number of repeating characters in the URL
    features.append(sum(url.count(char) for char in set(url)) - len(url))
    
    # 41. Length of the shortest word in the URL
    features.append(min([len(word) for word in url.split('/') if len(word) > 0], default=0))
    
    # 42. Length of the shortest word in the hostname
    features.append(min([len(word) for word in hostname.split('.') if len(word) > 0], default=0))
    
    # 43. Length of the shortest word in the path
    features.append(min([len(word) for word in parsed_url.path.split('/') if len(word) > 0], default=0))
    
    # 44. Length of the longest word in the URL
    features.append(max([len(word) for word in url.split('/') if len(word) > 0], default=0))
    
    # 45. Length of the longest word in the hostname
    features.append(max([len(word) for word in hostname.split('.') if len(word) > 0], default=0))
    
    # 46. Length of the longest word in the path
    features.append(max([len(word) for word in parsed_url.path.split('/') if len(word) > 0], default=0))
    
    # 47. Average word length in the raw URL
    features.append(np.mean([len(word) for word in url.split('/') if len(word) > 0]) if len(url.split('/')) > 0 else 0)
    
    # 48. Average word length in the hostname
    features.append(np.mean([len(word) for word in hostname.split('.') if len(word) > 0]) if len(hostname.split('.')) > 0 else 0)
    
    # 49. Average word length in the path
    features.append(np.mean([len(word) for word in parsed_url.path.split('/') if len(word) > 0]) if len(parsed_url.path.split('/')) > 0 else 0)
    
    # 50. Phishing hints (specific URL patterns indicative of phishing attempts)
    features.append(1 if 'login' in url or 'signin' in url else 0)
    
    # 51. Domain in brand name (e.g., brand-related domains)
    features.append(1 if 'brand' in hostname else 0)
    
    # 52. Brand in subdomain (e.g., 'secure.brand.com')
    features.append(1 if 'brand' in parsed_url.hostname.split('.')[0] else 0)
    
    # 53. Brand in path (e.g., 'login.brand.com')
    features.append(1 if 'brand' in parsed_url.path else 0)
    
    # 54. Suspicious top-level domain (TLD)
    features.append(1 if parsed_url.hostname.endswith('.xyz') else 0)
    
    # 55. Statistical report (page statistics like number of words, links, etc.)
    features.append(1 if 'report' in parsed_url.path else 0)
    
    # 56. Number of hyperlinks on the page
    features.append(url.count('<a href='))
    
    # 57. Ratio of internal hyperlinks (those pointing to the same domain)
    features.append(url.count('href="http') / len(url) if len(url) > 0 else 0)
    
    # 58. Ratio of external hyperlinks
    features.append(url.count('href="https') / len(url) if len(url) > 0 else 0)
    
    # 59. Presence of 'login' or 'account' in the path
    features.append(1 if 'login' in parsed_url.path or 'account' in parsed_url.path else 0)
    
    # 60. Path length relative to the total URL length
    features.append(len(parsed_url.path) / len(url) if len(url) > 0 else 0)
    
    # 61. Query parameters in the URL (indicating user input or form submission)
    features.append(1 if '?' in parsed_url.query else 0)
    
    # 62. URL contains query parameter 'username'
    features.append(1 if 'username' in parsed_url.query else 0)
    
    # 63. URL contains query parameter 'password'
    features.append(1 if 'password' in parsed_url.query else 0)
    
    # 64. Number of occurrences of 'javascript'
    features.append(url.count('javascript'))
    
    # 65. HTTPS connection status (secure connection)
    features.append(1 if parsed_url.scheme == 'https' else 0)
    
    # 66. SSL certificate presence check (indicating secure connection)
    features.append(1 if re.search(r'https://', url) else 0)
    
    # 67. Suspicious URL part (e.g., login forms with suspicious elements)
    features.append(1 if 'login' in parsed_url.path else 0)
    
    # 68. 'Signup' related part in the URL path
    features.append(1 if 'signup' in parsed_url.path else 0)
    
    # 69. URL contains 'paypal' (indicating payment-related links)
    features.append(1 if 'paypal' in url else 0)
    
    # 70. Payment gateway present in the URL
    features.append(1 if any(word in url for word in ['payment', 'checkout', 'order']) else 0)
    
    # 71. Suspicious URL path (e.g., unexpected directories)
    features.append(1 if 'admin' in parsed_url.path else 0)
    
    # 72. Presence of 'admin' in the URL
    features.append(1 if 'admin' in parsed_url.path else 0)
    
    # 73. Suspicious subdomain (e.g., login related)
    features.append(1 if 'login' in parsed_url.hostname else 0)
    
    # 74. Presence of IP addresses in the hostname
    features.append(1 if parsed_url.hostname.isdigit() else 0)
    
    # 75. DNS resolution of the URL
    features.append(1 if parsed_url.hostname else 0)
    
    # 76. Presence of suspicious patterns (e.g., 'beta', 'stage', etc.)
    features.append(1 if any(word in url for word in ['beta', 'stage']) else 0)
    
    # 77. URL path has more than 10 directories (indicating complex structure)
    features.append(1 if len(parsed_url.path.split('/')) > 10 else 0)
    
    # 78. Ratio of long URLs in the path
    features.append(sum(1 for part in parsed_url.path.split('/') if len(part) > 5) / len(parsed_url.path.split('/')) if len(parsed_url.path.split('/')) > 0 else 0)
    
    # 79. Short URL (URL length less than 30 characters)
    features.append(1 if len(url) < 30 else 0)
    
    # 80. Long URL (URL length more than 150 characters)
    features.append(1 if len(url) > 150 else 0)
    
    # 81. Presence of non-alphanumeric characters (e.g., '@', '-', '_')
    features.append(1 if re.search(r'[^a-zA-Z0-9]', url) else 0)
    
    # 82. Presence of digits in the URL path
    features.append(1 if any(c.isdigit() for c in parsed_url.path) else 0)
    
    # 83. Number of components in the path
    features.append(len(parsed_url.path.split('/')) if parsed_url.path else 0)
    
    # 84. URL contains 'secure' keyword
    features.append(1 if 'secure' in url else 0)
    
    # 85. URL contains 'http' keyword
    features.append(1 if 'http' in url else 0)
    
    # 86. Use of non-standard port in the URL
    features.append(1 if parsed_url.port and parsed_url.port != 80 and parsed_url.port != 443 else 0)
    
    # 87. Suspicious patterns in URL (e.g., 'download', 'exe', etc.)
    features.append(1 if any(pattern in url for pattern in ['download', 'exe']) else 0)
    
    return features

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment or default to 5000
    app.run(host='0.0.0.0', port=port)

