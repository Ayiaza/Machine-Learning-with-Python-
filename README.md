 # 🌸 Iris Flower Classifier — Random Forest ML Model

A beginner-friendly machine learning project that trains a Random Forest classifier on the classic Iris dataset to predict flower species with high accuracy — covering the full ML pipeline from data loading to model evaluation.

---

## 🛠️ Technologies Used

- **Python 3** — Core programming language
- **pandas** — Used to load and structure the dataset into a readable DataFrame
- **scikit-learn** — Provides the Iris dataset, train/test splitting, Random Forest model, and evaluation metrics
- **matplotlib** — Imported for potential data visualization support
- **RandomForestClassifier** — An ensemble learning model that combines multiple decision trees for accurate predictions
- **accuracy_score & classification_report** — Metrics used to measure how well the model performs

---

## 🔄 How It Works — Step by Step

1. **Load the dataset** — The built-in Iris dataset is loaded from scikit-learn and converted into a pandas DataFrame with proper column names
2. **Explore the data** — The first 5 rows of features (sepal length, sepal width, petal length, petal width) are printed for inspection
3. **Split the data** — The dataset is divided into 80% training and 20% testing using `train_test_split` with a fixed `random_state=42` for reproducibility
4. **Initialize the model** — A `RandomForestClassifier` is created with `random_state=42` to ensure consistent results across runs
5. **Train the model** — The model is fitted on the training data using `model.fit(X_train, y_train)`
6. **Make predictions** — The trained model predicts flower species for the unseen test data
7. **Evaluate performance** — `accuracy_score` calculates overall accuracy and `classification_report` breaks down precision, recall, and F1-score per class

---

## 📚 What I Learned

- How to build a complete machine learning pipeline in Python — from loading and splitting data to training and evaluating a classifier using scikit-learn
- How Random Forest works as an ensemble model, and how metrics like accuracy, precision, recall, and F1-score are used to measure real model performance

---

## ✅ Conclusion

This project demonstrates a clean, end-to-end machine learning workflow using one of the most popular algorithms in data science. It's a strong foundation for anyone stepping into the world of supervised learning and model evaluation.
----
## Run the Project
