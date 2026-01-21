# Wine Cultivar Origin Prediction - Model Development


import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Wine dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['cultivar'] = wine.target


# 2. Feature selection (choose any 6 allowed features)
features = [
    'alcohol',
    'malic_acid',
    'alcalinity_of_ash',
    'magnesium',
    'color_intensity',
    'proline'
]


target = 'cultivar'


X = df[features]
y = df[target]


# 3. Handle missing values (dataset has none, but included for completeness)
X = X.fillna(X.mean())


# 4. Feature scaling (MANDATORY)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# 6. Model implementation
model = SVC(kernel='rbf', probability=True)
model.fit(X_train, y_train)


# 7. Evaluation
y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# 8. Save model and scaler
joblib.dump(model, 'wine_cultivar_model.pkl')
joblib.dump(scaler, 'scaler.pkl')


print("Model and scaler saved successfully")