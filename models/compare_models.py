import json
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# -----------------------------
# Load Training Data
# -----------------------------
X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

print("=" * 70)
print("TRUSTLENS AI - MODEL COMPARISON")
print("=" * 70)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
}

results = []

best_model = None
best_model_name = ""
best_accuracy = 0
best_precision = 0
best_recall = 0
best_f1 = 0

# -----------------------------
# Train & Evaluate
# -----------------------------
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
        }
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, predictions))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = name
        best_precision = precision
        best_recall = recall
        best_f1 = f1

# -----------------------------
# Results Table
# -----------------------------
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df)

# -----------------------------
# Save Results
# -----------------------------
results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

joblib.dump(
    best_model,
    "models/model.pkl"
)
metrics = {
    "best_model": best_model_name,
    "accuracy": float(best_accuracy),
    "precision": float(best_precision),
    "recall": float(best_recall),
    "f1_score": float(best_f1)
}

with open("models/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Metrics saved successfully!")

print("\nBest Model Saved Successfully!")

print("\nBest Accuracy:", best_accuracy)

print("=" * 70)
print("PHASE 5 COMPLETED")
print("=" * 70)