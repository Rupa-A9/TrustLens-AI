"""
=========================================================
TrustLens AI
Phase 6 - Model Evaluation
=========================================================
"""

import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

OUTPUT_DIR = "outputs"


def create_output_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_files():
    model = joblib.load("models/model.pkl")

    X_test = joblib.load("models/X_test.pkl")
    y_test = joblib.load("models/y_test.pkl")

    with open("models/metrics.json") as f:
        metrics = json.load(f)

    return model, X_test, y_test, metrics


def save_classification_report(y_true, y_pred):

    report = classification_report(y_true, y_pred)

    with open(
        "outputs/classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    print("✔ Classification Report Saved")


def save_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)

    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig("outputs/confusion_matrix.png")

    plt.close()

    print("✔ Confusion Matrix Saved")


def save_accuracy_chart(metrics):

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Accuracy"],
        [metrics["accuracy"] * 100]
    )

    plt.ylim(0, 100)

    plt.ylabel("Percentage")

    plt.title("Best Model Accuracy")

    plt.tight_layout()

    plt.savefig("outputs/accuracy_chart.png")

    plt.close()

    print("✔ Accuracy Chart Saved")


def main():

    print("=" * 60)
    print("TRUSTLENS AI - MODEL EVALUATION")
    print("=" * 60)

    create_output_directory()

    model, X_test, y_test, metrics = load_files()

    predictions = model.predict(X_test)

    save_classification_report(
        y_test,
        predictions
    )

    save_confusion_matrix(
        model,
        X_test,
        y_test
    )

    save_accuracy_chart(
        metrics
    )

    print("\nBest Model :", metrics["best_model"])
    print("Accuracy   :", round(metrics["accuracy"] * 100, 2), "%")
    print("Precision  :", round(metrics["precision"] * 100, 2), "%")
    print("Recall     :", round(metrics["recall"] * 100, 2), "%")
    print("F1 Score   :", round(metrics["f1_score"] * 100, 2), "%")

    print("\nOutputs generated successfully.")

    print("=" * 60)
    print("PHASE 6 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()