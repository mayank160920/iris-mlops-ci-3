#!/usr/bin/env python
# coding: utf-8

# ## Simple Decision Tree model
# Build a Decision Tree model on iris data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import sys
import os
print("Python version:", sys.version)

DATASET_URI = "data.csv"
data = pd.read_csv(DATASET_URI)
print("Data read successfully.")
from mlflow.models import infer_signature
import mlflow

# MLflow configuration (can be overridden via environment variables)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:8100")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Iris Species Classification")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# set registry uri same as tracking by default (useful for tests that call set_registry_uri)
mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species
print("Data split into train and test sets.")


def execute_training_pipeline(hyperparams, X_train, y_train, X_test, y_test, registered_model_name="IrisDecisionTreeModel"):
    """Train DecisionTreeClassifier with given hyperparameters, evaluate and log to MLflow.

    Returns:
        model, metrics_dict
    """
    # Train
    model = DecisionTreeClassifier(**hyperparams)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = metrics.f1_score(y_test, y_pred, average='macro')

    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_params(hyperparams)
        mlflow.log_metrics(metrics_dict)

        # infer signature from training data and model predictions
        try:
            signature = infer_signature(X_train, model.predict(X_train))
        except Exception:
            signature = None

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="decision_tree_model",
            signature=signature,
            registered_model_name=registered_model_name,
        )

    print("Training pipeline executed and logged to MLflow.")

    # Save local model for compatibility
    try:
        if os.path.exists("model.joblib"):
            os.remove("model.joblib")
        joblib.dump(model, "model.joblib")
        print("Model saved locally to model.joblib")
    except Exception as e:
        print(f"Warning: failed to save local model.joblib: {e}")

    return model, metrics_dict


if __name__ == "__main__":
    print("Training Decision Tree model (with MLflow logging)...")
    hyperparams = {
        "criterion": "gini",
        "max_depth": 3,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 1,
    }
    model, metrics_dict = execute_training_pipeline(hyperparams, X_train, y_train, X_test, y_test)
    print('Final metrics:', metrics_dict)