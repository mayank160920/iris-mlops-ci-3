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

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)
X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
y_train = train.species
X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
y_test = test.species
print("Data split into train and test sets.")

print("Training Decision Tree model...")
mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
mod_dt.fit(X_train,y_train)
print("Model trained successfully.")

prediction=mod_dt.predict(X_test)
print('The accuracy of the Decision Tree is',"{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

print("Saving the model to model.joblib...")
# delete file if it exists (local)
if os.path.exists("model.joblib"):
    os.remove("model.joblib")
joblib.dump(mod_dt, "model.joblib")
print("Model saved successfully.")