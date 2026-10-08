import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Dataset from Scikit-Learn
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

print("=== DATASET OVERVIEW ===")
print(X.head())

# 2. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n=== DATASET SPLIT ===")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 3. Model Training using Decision Tree Classifier
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# 4. Model Prediction and Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n=== MODEL ACCURACY ===")
print(f"Accuracy Score: {accuracy * 100:.2f}%")

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 5. Feature Importance Analysis
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(importance)

# 6. Decision Tree Text Representation
print("\n=== DECISION TREE RULES ===")
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)