# app.py

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model pipeline
try:
    with open('churn_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("[ERROR] 'churn_model.pkl' not found. Please run model_training.py first.")
    model = None

# Define the main route to render the HTML page
@app.route('/')
def home():
    # This tells Flask to find 'index.html' in the 'templates' folder
    return render_template('index.html')

# Define the prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'The prediction model is not loaded.'}), 500

    try:
        # Get the customer data sent from the frontend
        data = request.get_json()
        
        # Create a pandas DataFrame from the incoming data
        input_df = pd.DataFrame([data])
        
        # Use the loaded pipeline to make a prediction
        # The pipeline automatically handles all the preprocessing (scaling, encoding, etc.)
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Extract the results
        result = int(prediction[0])
        probability_churn = float(prediction_proba[0][1])

        # Send the results back to the frontend in JSON format
        return jsonify({
            'prediction': result, # Will be 0 (No Churn) or 1 (Churn)
            'probability': round(probability_churn * 100, 2)
        })

    except Exception as e:
        print(f"[ERROR] An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 400

# This block allows you to run the server by executing "python app.py"
# if __name__ == '__main__':
    # debug=True will automatically reload the server when you save changes
    app.run(debug=True)
