from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data from the request
        data = request.get_json()

        # Extract features from the input data (make sure to get the correct feature names)
        features = np.array([data['length_url'], data['length_hostname'], data['ip'], data['nb_dots'],
                             data['nb_hyphens'], data['nb_at'], data['domain_age'], data['web_traffic'],
                             data['dns_record'], data['google_index'], data['page_rank']])

        # Reshape the input to match the expected shape of the model (1 sample, multiple features)
        features = features.reshape(1, -1)

        # Scale the input using the same scaler used during training
        scaled_features = scaler.transform(features)

        # Predict using the loaded model
        prediction = model.predict(scaled_features)

        # Return the result as JSON
        return jsonify({'prediction': 'phishing' if prediction[0] == 1 else 'legitimate'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment or default to 5000
    app.run(host='0.0.0.0', port=port)
