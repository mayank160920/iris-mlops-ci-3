"""
Week 8 - Data Poisoning Experiments on IRIS with MLflow
Reads dataset from: data/data.csv

Usage:
    python poisoning_experiments.py
"""

import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier


CSV_PATH = "data/data.csv"  # <= YOUR CSV FILE


def load_data(test_size: float = 0.2, random_state: int = 42):
    """
    Load dataset from CSV instead of sklearn.
    CSV must contain:
        sepal_length, sepal_width, petal_length, petal_width, species
    """

    df = pd.read_csv(CSV_PATH)

    # Encode species as numeric labels for sklearn
    df["species"] = df["species"].astype("category")
    y = df["species"].cat.codes.values  # numeric labels 0/1/2
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y, # type: ignore
    )
    return X_train, X_val, y_train, y_val


def poison_data(
    X_train: np.ndarray, ratio: float, random_state: int = 42
) -> np.ndarray:
    """Poison a fraction of rows in X_train by injecting random values."""

    if ratio <= 0.0:
        return X_train.copy()

    rng = np.random.default_rng(seed=random_state)
    X_poisoned = X_train.copy()

    n_samples, n_features = X_poisoned.shape
    n_poison = max(1, int(n_samples * ratio))

    # Choose which rows to corrupt
    poisoned_indices = rng.choice(n_samples, size=n_poison, replace=False)

    # Range per feature based on original (clean) range
    feature_mins = X_train.min(axis=0)
    feature_maxs = X_train.max(axis=0)

    for j in range(n_features):
        low, high = feature_mins[j], feature_maxs[j]
        random_values = rng.uniform(low=low, high=high, size=n_poison)
        X_poisoned[poisoned_indices, j] = random_values

    return X_poisoned


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    corruption_ratio: float,
):
    """Train on poisoned data, evaluate on clean data, log to MLflow."""

    run_name = f"corruption_{int(corruption_ratio * 100)}pct"

    with mlflow.start_run(run_name=run_name):
        # Log params
        mlflow.log_param("corruption_ratio", corruption_ratio)

        X_poisoned = poison_data(X_train, corruption_ratio)
        n_samples = X_train.shape[0]
        n_poison = int(n_samples * corruption_ratio)

        mlflow.log_param("n_train_samples", n_samples)
        mlflow.log_param("n_poisoned_samples", n_poison)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_poisoned, y_train)

        # Validation metrics
        y_pred = model.predict(X_val)
        accuracy = float(accuracy_score(y_val, y_pred))
        precision = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
        recall = float(recall_score(y_val, y_pred, average="macro", zero_division=0))

        mlflow.log_metric("val_accuracy", accuracy)
        mlflow.log_metric("val_precision_macro", precision)
        mlflow.log_metric("val_recall_macro", recall)

        # Save model artifact
        mlflow.sklearn.log_model(model, name="model", input_example=X_val[:5])

        print(
            f"[{run_name}] accuracy={accuracy:.4f}, "
            f"precision={precision:.4f}, recall={recall:.4f}"
        )


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("week8_iris_data_poisoning")

    X_train, X_val, y_train, y_val = load_data()

    corruption_levels = [0.0, 0.05, 0.10, 0.50]

    for level in corruption_levels:
        train_and_evaluate(X_train, y_train, X_val, y_val, level)


if __name__ == "__main__":
    main()
