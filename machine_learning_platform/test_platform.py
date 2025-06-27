import requests
import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

BASE_URL = 'http://localhost:5000'
TEST_CREDENTIALS = ('testuser', 'testpass123')

def test_registration():
    response = requests.post(
        f'{BASE_URL}/register',
        json={'username': TEST_CREDENTIALS[0], 'password': TEST_CREDENTIALS[1]}
    )
    print("Registration test:", response.json())
    return response.status_code == 200

def test_preloaded_resources():
    # Test preloaded models endpoint
    models_response = requests.get(f'{BASE_URL}/get_preloaded_models')
    print("Preloaded models:", models_response.json())
    
    # Test preloaded datasets endpoint
    datasets_response = requests.get(f'{BASE_URL}/get_preloaded_datasets')
    print("Preloaded datasets:", datasets_response.json())
    
    return models_response.status_code == 200 and datasets_response.status_code == 200

def test_model_upload():
    # Create a simple model for testing
    model = LogisticRegression()
    model_file = 'test_model.pkl'
    joblib.dump(model, model_file)
    
    files = {'model': open(model_file, 'rb')}
    data = {'model_id': 'test_model_1'}
    
    response = requests.post(
        f'{BASE_URL}/upload_model',
        files=files,
        data=data,
        auth=TEST_CREDENTIALS
    )
    print("Model upload test:", response.json())
    return response.status_code == 200

def test_batch_prediction():
    # Test batch prediction with sample data
    test_data = {
        'model_id': 'test_model_1',
        'version_id': '1',  # Use the version ID received from upload
        'input_data': [[1, 2, 3, 4], [5, 6, 7, 8]]
    }
    
    response = requests.post(
        f'{BASE_URL}/batch_predict',
        json=test_data,
        auth=TEST_CREDENTIALS
    )
    print("Batch prediction test:", response.json())
    return response.status_code == 200

def test_monitoring():
    response = requests.get(
        f'{BASE_URL}/monitor',
        auth=TEST_CREDENTIALS
    )
    print("Monitoring stats:", response.json())
    return response.status_code == 200

def run_all_tests():
    tests = {
        "Registration": test_registration,
        "Preloaded Resources": test_preloaded_resources,
        "Model Upload": test_model_upload,
        "Batch Prediction": test_batch_prediction,
        "Monitoring": test_monitoring
    }
    
    results = {}
    for test_name, test_func in tests.items():
        try:
            success = test_func()
            results[test_name] = "PASS" if success else "FAIL"
        except Exception as e:
            results[test_name] = f"ERROR: {str(e)}"
    
    print("\nTest Results:")
    print("-------------")
    for test_name, result in results.items():
        print(f"{test_name}: {result}")

if __name__ == "__main__":
    run_all_tests() 