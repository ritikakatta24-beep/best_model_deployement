import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# -----------------------------
# Setup Flask
# -----------------------------
app = Flask(__name__, template_folder='.')  # look for HTML files in the current folder

# Load your model
MODEL_FILE = "best_model_linear_regression.joblib"
loaded = joblib.load(MODEL_FILE)

if isinstance(loaded, dict) and 'model' in loaded:
    model = loaded['model']
else:
    model = loaded

print("Model type:", type(model))  # should show sklearn Pipeline

# Expected features
numeric_features = ['ENGINE_SIZE','CYLINDERS','FUEL_CONSUMPTION']
categorical_features = ['MAKE','MODEL','VEHICLE_CLASS','TRANSMISSION','FUEL']
expected_columns = numeric_features + categorical_features

# -----------------------------
# Serve HTML page
# -----------------------------
@app.route('/')
def home_page():
    return render_template('index.html')  # render index.html

# -----------------------------
# Health check endpoint
# -----------------------------
@app.route('/health')
def health():
    return jsonify({"healthy": model is not None}), 200 if model else 503

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    
    if data is None or 'features' not in data:
        return jsonify({'error': "Missing 'features' key", 'prediction': None}), 400

    features = data['features']

    # Convert single row to list
    if isinstance(features, dict):
        features = [features]

    X_new = pd.DataFrame(features)
    print("Incoming data:\n", X_new)  # DEBUG

    # Align columns
    X_new = X_new.reindex(columns=expected_columns)
    print("Aligned data for model:\n", X_new)  # DEBUG

    try:
        preds = model.predict(X_new)
        print("Predictions:", preds)  # DEBUG
        return jsonify({'prediction': preds.tolist(), 'error': None}), 200
    except Exception as e:
        print("Prediction error:", e)  # DEBUG
        return jsonify({'prediction': None, 'error': str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
