import os
import pandas as pd

def test_data_file_exists():
    assert os.path.exists("data.csv"), "data.csv file not found."
    print("data.csv file exists.")

def test_data_has_expected_columns():
    df = pd.read_csv("data.csv")
    expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    assert all(col in df.columns for col in expected_columns)
    print("data.csv has expected columns.")
