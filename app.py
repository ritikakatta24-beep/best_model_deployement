import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


app = Flask(__name__, template_folder='.')  

MODEL_FILE = "best_model_linear_regression.joblib"
loaded = joblib.load(MODEL_FILE)

if isinstance(loaded, dict) and 'model' in loaded:
    model = loaded['model']
else:
    model = loaded

print("Model type:", type(model)) 


numeric_features = ['ENGINE_SIZE','CYLINDERS','FUEL_CONSUMPTION']
categorical_features = ['MAKE','MODEL','VEHICLE_CLASS','TRANSMISSION','FUEL']
expected_columns = numeric_features + categorical_features


@app.route('/')
def home_page():
    return render_template('index.html')  


@app.route('/health')
def health():
    return jsonify({"healthy": model is not None}), 200 if model else 503

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    
    if data is None or 'features' not in data:
        return jsonify({'error': "Missing 'features' key", 'prediction': None}), 400

    features = data['features']

    if isinstance(features, dict):
        features = [features]

    X_new = pd.DataFrame(features)
    print("Incoming data:\n", X_new)  

    
    X_new = X_new.reindex(columns=expected_columns)
    print("Aligned data for model:\n", X_new)  

    try:
        preds = model.predict(X_new)
        print("Predictions:", preds)  
        return jsonify({'prediction': preds.tolist(), 'error': None}), 200
    except Exception as e:
        print("Prediction error:", e) 
        return jsonify({'prediction': None, 'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
