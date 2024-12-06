from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('phishing_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load feature names from the features file
with open('features.txt', 'r') as f:
    feature_columns = [line.strip() for line in f.readlines()]

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input JSON data from the request
        data = request.get_json()

        # Validate input
        if not isinstance(data, dict):
            raise ValueError("Input data must be a JSON object.")

        # Create a DataFrame with a single row, using the feature names
        # Fill missing features with 0
        input_data = pd.DataFrame([{key: data.get(key, 0) for key in feature_columns}])

        # Scale the input using the same scaler used during training
        scaled_features = scaler.transform(input_data)

        # Predict using the loaded model
        prediction = model.predict(scaled_features)

        # Return the result as JSON
        return jsonify({'prediction': 'phishing' if prediction[0] == 1 else 'legitimate'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment or default to 5000
    app.run(host='0.0.0.0', port=port)
