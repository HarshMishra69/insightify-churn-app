# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the churn model
try:
    with open('churn_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Churn model loaded successfully!")
except FileNotFoundError:
    print("[ERROR] 'churn_model.pkl' not found. Please run model_training.py first.")
    model = None

# Get the API Key from the environment variable set in Render
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/config')
def config():
    return jsonify({'apiKey': GEMINI_API_KEY})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Prediction model not loaded on the server.'}), 500
        
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        result = int(prediction[0])
        probability_churn = float(prediction_proba[0][1])
        
        return jsonify({
            'prediction': result,
            'probability': round(probability_churn * 100, 2)
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400

# Note: The if __name__ == '__main__': block is removed for deployment.
# The Gunicorn server specified in the Render start command will run the app.
