# Graded Assignment — Week 6
## File details

This repository contains code and artifacts used for training, packaging, and testing an Iris classifier. Important files and folders:

- **`data.csv`**
  The dataset used for training and tests (CSV format).

- **`data.csv.dvc`**
  DVC metadata for tracking `data.csv` (if DVC is used in the course environment).

- **`train.py`**
  Script that reads the dataset, trains the model, and saves the trained artifact locally.

- **`train.ipynb`**
  Jupyter notebook version of the training pipeline with exploratory steps and visualizations.

- **`setup.ipynb`**
  Notebook with setup commands and environment preparation used in the assignment.

- **`instructions.ipynb`**
  Notebook containing assignment instructions, notes, or guided steps for the exercise.

- **`iris_fastapi.py`**
  A small FastAPI app to serve the trained model for inference (HTTP endpoints for predict/health).

- **`load_model_from_mlflow.py`**
  Helper script to load a model from MLflow (used when the model is stored in an MLflow tracking server).

- **`Dockerfile`**
  Docker image definition to containerize the app (model server or inference service).

- **`k8s/`**
  Kubernetes manifests used to deploy the containerized service. Example files:
  - `deployment.yaml` — Deployment spec for the model service
  - `service.yaml` — Service spec exposing the deployment

- **`requirements.txt`**
  Python dependencies required to run the scripts and app.

- **`README.md`**
  This file — describes project structure and purpose.

- **`tests/`**
  Unit tests and simple checks. Key tests include:
  - `tests/test_data.py` — data integrity / schema checks
  - `tests/test_model.py` — model training / inference sanity checks
