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
        # Get the input JSON data
        data = request.get_json()

        # Extract features and fill missing ones with defaults (0)
        input_features = pd.DataFrame([data], columns=feature_columns).fillna(0)

        print("Raw Input Features:", input_features)

        # Scale the input
        scaled_features = scaler.transform(input_features)

        print("Scaled Features:", scaled_features)

        # Predict
        prediction = model.predict(scaled_features)

        # Debug prediction output
        print("Prediction:", prediction)

        return jsonify({'prediction': 'phishing' if prediction[0] == 1 else 'legitimate'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get the port from the environment or default to 5000
    app.run(host='0.0.0.0', port=port)
