# Machine Learning Model Deployment Platform

## Overview

A Flask-based platform for uploading, versioning, training, and deploying machine learning models with authentication, monitoring, and preloaded resources.

## Features

- Model upload, versioning, and export
- Dataset upload and preloaded datasets
- Pre-trained models
- Batch and single prediction endpoints
- Performance metrics and monitoring
- User authentication (basic auth)

## Setup

1. **Clone the repository**
   ```sh
   git clone <your-repo-url>
   cd <repo-folder>
   ```
2. **Create a virtual environment (recommended)**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
4. **(Optional) Install test dependencies**
   ```sh
   pip install -r test_requirements.txt
   ```
5. **Run the server**
   ```sh
   python app.py
   ```

## Usage

- Register a user: `POST /register` with JSON `{"username": ..., "password": ...}`
- Upload a model: `POST /upload_model` (auth required, multipart form)
- Upload a dataset: `POST /upload_dataset` (auth required, multipart form)
- Train a model: `POST /train_model` (auth required, JSON)
- Predict: `POST /predict` (auth required, JSON)
- Batch predict: `POST /batch_predict` (auth required, JSON)
- Get preloaded models: `GET /get_preloaded_models`
- Get preloaded datasets: `GET /get_preloaded_datasets`
- Monitor: `GET /monitor` (auth required)
- Export model: `GET /export_model` (auth required, query params)
- Get model metrics: `GET /model_metrics` (auth required, query params)

## Preloaded Models & Datasets

Place your `.pkl` models in `preloaded_models/` and datasets in `preloaded_datasets/` as CSV files. Some are included by default (e.g., iris, diabetes).

## Testing

Run the test suite:

```sh
python test_platform.py
```

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

## License

MIT
