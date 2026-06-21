import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("--- Step 1: Loading the Dataset ---")
# Load the classic Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Features (First 5 rows):")
print(X.head())
print("-" * 40)

print("\n--- Step 2: Splitting Data into Train & Test ---")
# Split data: 80% for training, 20% for testing to evaluate performance fairly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")
print("-" * 40)

print("\n--- Step 3: Training the Random Forest Model ---")
# Initialize the model with a fixed random state for reproducibility
model = RandomForestClassifier(random_state=42)

# Train (fit) the model
model.fit(X_train, y_train)
print("Model training complete!")
print("-" * 40)

print("\n--- Step 4: Evaluating the Model ---")
# Predict the species for the test data
y_pred = model.predict(X_test)

# Calculate and display the metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("-" * 40)