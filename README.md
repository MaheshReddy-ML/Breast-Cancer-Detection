# 🧬 Breast Cancer Detection using Logistic Regression (From Scratch)

This project implements a **logistic regression model from scratch** to predict whether a tumor is **malignant (cancerous)** or **benign (non-cancerous)**. No external ML libraries like `scikit-learn` or `TensorFlow` are used — just **NumPy**, **manual gradient descent**, and core math logic.

---

## 🔍 Overview

- 📂 Dataset: Breast Cancer Wisconsin (Diagnostic) - [Kaggle Link](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- 🛠️ ML Algorithm: Logistic Regression
- 🔁 Optimizer: Gradient Descent (built manually)
- 🧮 Accuracy Achieved: **~92%**
- 🧠 Input: 6 numerical features from the dataset
- 🚫 No ML libraries like `sklearn` were used
- 📈 Libraries Used: `NumPy`, `Pandas`, `Matplotlib`

---

## 🧪 Features Used

- `radius_mean`: Mean of distances from center to points on the perimeter
- `texture_mean`: Standard deviation of gray-scale values
- `perimeter_mean`: Perimeter of the tumor
- `area_mean`: Area of the tumor
- `concavity_mean`: Severity of concave portions
- `concave points_mean`: Number of concave portions
## 💡 How to Use

Run the script, and when prompted, enter the 6 tumor-related measurements. The model will output whether the tumor is:

✅ **Benign (Non-Cancerous)**  
🚨 **Malignant (Cancerous)**
---

## 🧪 Try It Yourself (Sample Input)

Wanna test the model right away? Here's a sample input you can try when the program prompts you:

Enter radius_mean: 14.5
Enter texture_mean: 20.3
Enter perimeter_mean: 96.5
Enter area_mean: 680.0
Enter concavity_mean: 0.15
Enter concave points_mean: 0.07

🧬 **Prediction: Malignant (Cancerous)** 🚨  
*(based on model’s decision boundary)*

---

Feel free to change the values and test different tumor profiles!
