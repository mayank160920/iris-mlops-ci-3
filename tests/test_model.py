import joblib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

def test_model_accuracy():
    model = joblib.load("model.joblib")
    data = pd.read_csv("data.csv")
    
    _, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test['species']
    
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(predictions, y_test)
    
    with open("report.txt", "w") as f:
        f.write(f"Model Accuracy: {accuracy:.3f}\n")

    assert accuracy > 0.9, f"Model accuracy {accuracy:.2f} is below threshold."
    print(f"Model accuracy {accuracy:.3f} is above threshold.")
