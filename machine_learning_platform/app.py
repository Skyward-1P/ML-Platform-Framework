from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datetime import datetime
import uuid
import os
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
auth = HTTPBasicAuth()
executor = ThreadPoolExecutor(4)

# Extended storage with versioning and performance metrics
models = {}  # {model_id: {versions: {version_id: {model: ..., metrics: ...}}}}
datasets = {}
users = {'admin': generate_password_hash('admin123')}

# Extended pre-loaded models
preloaded_models = {
    'logistic_regression': joblib.load('preloaded_models/logistic_regression.pkl'),
    'random_forest': joblib.load('preloaded_models/random_forest.pkl'),
    'svm_classifier': joblib.load('preloaded_models/svm_classifier.pkl'),
    'xgboost_classifier': joblib.load('preloaded_models/xgboost_classifier.pkl'),
    'knn_classifier': joblib.load('preloaded_models/knn_classifier.pkl')
}

# Extended pre-loaded datasets
preloaded_datasets = {
    'iris': pd.read_csv('preloaded_datasets/iris.csv'),
    'diabetes': pd.read_csv('preloaded_datasets/diabetes.csv'),
    'wine': pd.read_csv('preloaded_datasets/wine.csv'),
    'breast_cancer': pd.read_csv('preloaded_datasets/breast_cancer.csv'),
    'mnist_sample': pd.read_csv('preloaded_datasets/mnist_sample.csv')
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username

@app.route('/register', methods=['POST'])
def register():
    username = request.json.get('username')
    password = request.json.get('password')
    if username in users:
        return jsonify({"error": "User already exists"}), 400
    users[username] = generate_password_hash(password)
    return jsonify({"status": "success", "username": username})

@app.route('/upload_model', methods=['POST'])
@auth.login_required
def upload_model():
    model_file = request.files['model']
    model_id = request.form['model_id']
    version_id = str(uuid.uuid4())
    
    if model_id not in models:
        models[model_id] = {'versions': {}, 'created_at': datetime.now()}
    
    models[model_id]['versions'][version_id] = {
        'model': joblib.load(model_file),
        'timestamp': datetime.now()
    }
    return jsonify({"status": "success", "model_id": model_id, "version_id": version_id})

@app.route('/get_preloaded_models', methods=['GET'])
def get_preloaded_models():
    return jsonify({"preloaded_models": list(preloaded_models.keys())})

@app.route('/get_preloaded_datasets', methods=['GET'])
def get_preloaded_datasets():
    return jsonify({"preloaded_datasets": list(preloaded_datasets.keys())})

@app.route('/monitor', methods=['GET'])
@auth.login_required
def monitor():
    stats = {
        'total_models': len(models),
        'total_datasets': len(datasets),
        'model_versions': sum(len(m['versions']) for m in models.values()),
        'active_users': len(users)
    }
    return jsonify(stats)

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    dataset_file = request.files['dataset']
    dataset_id = request.form['dataset_id']
    datasets[dataset_id] = pd.read_csv(dataset_file)
    return jsonify({"status": "success", "dataset_id": dataset_id})

@app.route('/train_model', methods=['POST'])
def train_model():
    data = request.json
    model_id = data['model_id']
    dataset_id = data['dataset_id']
    
    model = models[model_id]['versions'][data['version_id']]['model']
    dataset = datasets[dataset_id]
    
    # Assuming last column is target
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    
    model.fit(X, y)
    return jsonify({"status": "success", "model_id": model_id})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    model_id = data['model_id']
    input_data = data['input_data']
    
    model = models[model_id]['versions'][data['version_id']]['model']
    prediction = model.predict([input_data])
    return jsonify({"prediction": prediction.tolist()})

@app.route('/batch_predict', methods=['POST'])
@auth.login_required
def batch_predict():
    data = request.json
    model_id = data['model_id']
    version_id = data['version_id']
    input_data = data['input_data']
    
    def process_prediction(model, data):
        return model.predict(data).tolist()
    
    model = models[model_id]['versions'][version_id]['model']
    predictions = list(executor.submit(process_prediction, model, input_data).result())
    return jsonify({"predictions": predictions})

@app.route('/model_metrics', methods=['GET'])
@auth.login_required
def get_model_metrics():
    model_id = request.args.get('model_id')
    version_id = request.args.get('version_id')
    
    if model_id not in models or version_id not in models[model_id]['versions']:
        return jsonify({"error": "Model or version not found"}), 404
    
    metrics = models[model_id]['versions'][version_id].get('metrics', {})
    return jsonify({"model_id": model_id, "version_id": version_id, "metrics": metrics})

@app.route('/export_model', methods=['GET'])
@auth.login_required
def export_model():
    model_id = request.args.get('model_id')
    version_id = request.args.get('version_id')
    
    if model_id not in models or version_id not in models[model_id]['versions']:
        return jsonify({"error": "Model or version not found"}), 404
    
    filename = f"exported_models/{model_id}_{version_id}.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(models[model_id]['versions'][version_id]['model'], filename)
    return jsonify({"status": "success", "download_link": filename})

if __name__ == '__main__':
    app.run(debug=True, threaded=True) 