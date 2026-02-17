# main_logistic.py
# ML Classification Project: Water Potability (SDG 6)
# Author: Anu Juliet

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
data = pd.read_csv('data/water_potability.csv')
print("Dataset Info:\n", data.info())
print("\nFirst 5 rows:\n", data.head())

# -----------------------------
# 2. Handle Missing Values
# -----------------------------
data.fillna(data.mean(), inplace=True)
print("\nMissing values after filling:\n", data.isnull().sum())

# -----------------------------
# 3. Prepare Features and Target
# -----------------------------
X = data.drop('Potability', axis=1)
y = data['Potability']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Train Logistic Regression
# -----------------------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluate Model on Test Set
# -----------------------------
y_pred = log_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation on Test Set ---")
print(f"Accuracy: {acc:.3f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
