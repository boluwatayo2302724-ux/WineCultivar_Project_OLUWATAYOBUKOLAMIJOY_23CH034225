# model_building.ipynb

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load Wine dataset
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["cultivar"] = data.target

# Select only allowed features
selected_features = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "flavanoids",
    "color_intensity"
]

X = df[selected_features]
y = df["cultivar"]

# Check missing values
print(df.isnull().sum())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (mandatory)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model & scaler
os.makedirs("model", exist_ok=True)
joblib.dump(model, "wine_cultivar_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")
