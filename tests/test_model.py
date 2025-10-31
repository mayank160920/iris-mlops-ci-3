import os
import sys
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import mlflow

# Define the model name used in the registry
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "IrisDecisionTreeModel")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8100")

def test_model_accuracy():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    # Load the latest version of the registered model
    try:
        latest_model_info = client.get_latest_versions(REGISTERED_MODEL_NAME)[0]
        print(f"Loaded model '{REGISTERED_MODEL_NAME}' version {latest_model_info.version} from registry.")
    except Exception as e:
        sys.exit(f"❌ Failed to get latest model version for '{REGISTERED_MODEL_NAME}': {e}")

    # Load the model from MLflow Model Registry
    try:
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/{latest_model_info.version}")
    except Exception as e:
        sys.exit(f"❌ Failed to load model '{REGISTERED_MODEL_NAME}/{latest_model_info.version}': {e}")

    # Check if test data file exists
    if not os.path.exists("data.csv"):
        sys.exit("❌ Missing test data file: data.csv")

    # Load test data
    data = pd.read_csv("data.csv")
    _, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']
    
    # Make predictions and calculate accuracy
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(predictions, y_test)
    report = str(metrics.classification_report(y_test, predictions,output_dict=True))
    model_info = f"Model: {REGISTERED_MODEL_NAME}, Version: {latest_model_info.version}"

    # Save accuracy report to a markdown file (so CI can include it nicely)
    REPORT_FILE = os.getenv("MODEL_REPORT_FILE", "model_report.md")
    report_dict = metrics.classification_report(y_test, predictions, output_dict=True)

    # Build a markdown table for per-class metrics
    lines = []
    lines.append(f"# {model_info}\n\n")
    lines.append(f"**Model Accuracy:** {accuracy:.3f}\n\n")
    lines.append("### Classification Report\n\n")
    lines.append("| class | precision | recall | f1-score | support |\n")
    lines.append("|---|---:|---:|---:|---:|\n")

    # Write class rows first (skip aggregate keys like accuracy / averages)
    for label, vals in report_dict.items():
        if label in ("accuracy", "macro avg", "weighted avg", "micro avg"):
            continue
        if isinstance(vals, dict):
            precision = vals.get("precision", 0.0)
            recall = vals.get("recall", 0.0)
            f1 = vals.get("f1-score", 0.0)
            support = int(vals.get("support", 0))
            lines.append(f"| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {support} |\n")

    # Add aggregate rows
    lines.append("\n")
    if "macro avg" in report_dict:
        m = report_dict["macro avg"]
        lines.append(f"**Macro avg** — precision: {m.get('precision', 0.0):.3f}, recall: {m.get('recall', 0.0):.3f}, f1: {m.get('f1-score', 0.0):.3f}\n\n")
    if "weighted avg" in report_dict:
        w = report_dict["weighted avg"]
        lines.append(f"**Weighted avg** — precision: {w.get('precision', 0.0):.3f}, recall: {w.get('recall', 0.0):.3f}, f1: {w.get('f1-score', 0.0):.3f}\n\n")
    if "accuracy" in report_dict:
        lines.append(f"**Overall accuracy:** {report_dict.get('accuracy'):.3f}\n\n")

    with open(REPORT_FILE, "w") as f:
        f.writelines(lines)

    assert accuracy > 0.9, f"Model accuracy {accuracy:.2f} is below threshold."
    print(f"Model accuracy {accuracy:.3f} is above threshold.")
