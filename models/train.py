import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# -------------------------------
# Create model directory
# -------------------------------
os.makedirs("models", exist_ok=True)

# -------------------------------
# Load processed dataset
# -------------------------------
df = pd.read_csv("data/processed_reviews.csv")

# Remove missing values
df = df.dropna(subset=["clean_review"])

# -------------------------------
# Encode Labels
# OR = Original (Real)
# CG = Computer Generated (Fake)
# -------------------------------
label_mapping = {
    "OR": 0,
    "CG": 1
}

df["label"] = df["label"].map(label_mapping)

print("=" * 60)
print("TRUSTLENS AI - FEATURE ENGINEERING")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nEncoded Labels:")
print(df["label"].value_counts())

# -------------------------------
# Features and Target
# -------------------------------
X = df["clean_review"]
y = df["label"]

# -------------------------------
# Train Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -------------------------------
# TF-IDF Vectorizer
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("\nTF-IDF Completed")

print("Training Shape :", X_train_vectorized.shape)
print("Testing Shape  :", X_test_vectorized.shape)

# -------------------------------
# Save Vectorizer
# -------------------------------
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nVectorizer saved successfully!")

# -------------------------------
# Save Train/Test Data
# -------------------------------
joblib.dump(X_train_vectorized, "models/X_train.pkl")
joblib.dump(X_test_vectorized, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("Training data saved successfully!")

print("\n" + "=" * 60)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("=" * 60)