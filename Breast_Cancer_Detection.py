"""
🧬 Breast Cancer Detection using Logistic Regression (From Scratch)

This script implements a logistic regression classifier from scratch to predict whether a tumor is malignant or benign.
It uses manual gradient descent, no ML libraries like scikit-learn, and processes the Breast Cancer Wisconsin dataset.

Author: Mahesh Reddy 
Accuracy Achieved: ~92%
Dataset: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Encode diagnosis: M = 1 (Malignant), B = 0 (Benign)
df["diagnosis"] = df["diagnosis"].map({'M': 1, 'B': 0})

# Select features
selected_features = [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "concavity_mean", "concave points_mean"
]

# Prepare inputs
X = df[selected_features].values
y = df["diagnosis"].values.reshape(-1, 1)

# Feature scaling
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Initialize parameters
m, n = X.shape
weights = np.zeros((n, 1))
bias = 0
lr = 0.01
epochs = 1000

# Gradient Descent
for epoch in range(epochs):
    z = np.dot(X, weights) + bias
    pred = sigmoid(z)
    cost = -(1/m) * np.sum(y * np.log(pred + 1e-9) + (1 - y) * np.log(1 - pred + 1e-9))
    dw = (1/m) * np.dot(X.T, (pred - y))
    db = (1/m) * np.sum(pred - y)
    weights -= lr * dw
    bias -= lr * db
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Cost: {cost:.4f}")

# Prediction function
def predict(x, weights, bias):
    z = np.dot(x, weights) + bias
    probs = sigmoid(z)
    return (probs > 0.5).astype(int)

# Evaluate model
y_pred = predict(X, weights, bias)
accuracy = np.mean(y_pred == y) * 100
print(f"\n✅ Final Accuracy: {accuracy:.2f}%")

# --- Tumor Symptom Input Assistant ---
print("\n--- Tumor Symptom Input Assistant ---")
print("We'll ask you to enter 6 tumor properties for prediction.\n")

print("🔵 radius_mean:")
print("Average size of the tumor cell nucleus.")
print("🔹 Small: 8–10, Medium: 12–15, Large: 18+ (suspicious)\n")
radius = float(input("Enter radius_mean: "))

print("\n🟣 texture_mean:")
print("Variation in shading across tumor cells.")
print("🔹 Low: 10–12, Medium: 15–20, High: 20+ (riskier)\n")
texture = float(input("Enter texture_mean: "))

print("\n🔶 perimeter_mean:")
print("Size of the tumor cell boundary (in pixels).")
print("🔹 Short: 50–70, Medium: 80–100, Long: 110+ (possibly malignant)\n")
perimeter = float(input("Enter perimeter_mean: "))

print("\n🟩 area_mean:")
print("Total area covered by tumor cells.")
print("🔹 Small: 400–600, Medium: 700–900, Large: 1000+ (often malignant)\n")
area = float(input("Enter area_mean: "))

print("\n🟠 concavity_mean:")
print("Measures how deeply the tumor cell edges curve inward.")
print("🔹 Low: <0.05, Medium: 0.05–0.15, High: >0.15\n")
concavity = float(input("Enter concavity_mean: "))

print("\n🟡 concave points_mean:")
print("Points where the cell edge curves inward (like dimples).")
print("🔹 Low: <0.03, Medium: 0.03–0.08, High: >0.08\n")
concave_points = float(input("Enter concave points_mean: "))

# Standardize user input
means = df[selected_features].mean()
stds = df[selected_features].std()

input_features = np.array([
    radius, texture, perimeter, area, concavity, concave_points
])
input_scaled = (input_features - means.values) / stds.values
x_input = input_scaled.reshape(1, -1)

# Prediction Time 😁😁
prediction = predict(x_input, weights, bias)


if prediction[0][0] == 1:
    print("\n🧬 Prediction: Malignant (Cancerous) 🚨")
else:
    print("\n🧬 Prediction: Benign (Non-Cancerous) ✅")
