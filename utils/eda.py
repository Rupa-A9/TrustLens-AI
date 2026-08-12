import os
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
from wordcloud import WordCloud

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("data/processed_reviews.csv")

# Remove rows with missing values in clean_review
df = df.dropna(subset=["clean_review"])

# Convert clean_review to string
df["clean_review"] = df["clean_review"].astype(str)

print("=" * 60)
print("TRUSTLENS AI - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLabel Distribution:")
print(df["label"].value_counts())

print("\nRating Distribution:")
print(df["rating"].value_counts().sort_index())

print("\nTop 10 Categories:")
print(df["category"].value_counts().head(10))

# ----------------------------
# Label Distribution
# ----------------------------
plt.figure(figsize=(6, 5))

df["label"].value_counts().plot(
    kind="bar",
    color=["#4CAF50", "#F44336"]
)

plt.title("Fake vs Genuine Reviews")
plt.xlabel("Label")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/label_distribution.png")
plt.close()

# ----------------------------
# Rating Distribution
# ----------------------------
plt.figure(figsize=(6, 5))

df["rating"].value_counts().sort_index().plot(
    kind="bar",
    color="orange"
)

plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig("outputs/rating_distribution.png")
plt.close()

# ----------------------------
# Category Distribution
# ----------------------------
plt.figure(figsize=(12, 6))

df["category"].value_counts().head(10).plot(
    kind="bar",
    color="steelblue"
)

plt.title("Top 10 Product Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("outputs/category_distribution.png")
plt.close()

# ----------------------------
# Missing Values
# ----------------------------
plt.figure(figsize=(8, 5))
msno.matrix(df)

plt.savefig("outputs/missing_values.png")
plt.close()

# ----------------------------
# Review Length Distribution
# ----------------------------
df["review_length"] = df["clean_review"].str.len()

plt.figure(figsize=(8, 5))

plt.hist(
    df["review_length"],
    bins=40,
    edgecolor="black"
)

plt.title("Review Length Distribution")
plt.xlabel("Number of Characters")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("outputs/review_length_distribution.png")
plt.close()

# ----------------------------
# Word Cloud
# ----------------------------

# Use a sample to avoid memory issues
sample_reviews = df["clean_review"].sample(
    n=min(5000, len(df)),
    random_state=42
)

text = " ".join(sample_reviews)

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white",
    max_words=300
).generate(text)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout()

plt.savefig("outputs/wordcloud.png")
plt.close()
print("\n" + "=" * 60)
print("✅ EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")

files = [
    "label_distribution.png",
    "rating_distribution.png",
    "category_distribution.png",
    "missing_values.png",
    "review_length_distribution.png",
    "wordcloud.png",
]

for file in files:
    print(f"✔ outputs/{file}")