import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (runs only the first time)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load dataset
df = pd.read_csv("data/fake_reviews_dataset.csv")

print("Original Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Find the review column automatically
possible_review_columns = [
    "review",
    "text",
    "text_",
    "reviewText",
    "content",
    "review_content"
]

review_column = None

for col in possible_review_columns:
    if col in df.columns:
        review_column = col
        break

if review_column is None:
    raise ValueError(
        f"No review column found!\nAvailable columns: {df.columns.tolist()}"
    )

print(f"\nUsing review column: {review_column}")

# NLP setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Remove missing values
df = df.dropna()

# Create cleaned review column
df["clean_review"] = df[review_column].apply(clean_text)

print("\nSample:")
print(df[[review_column, "clean_review"]].head())

# Save processed dataset
df.to_csv("data/processed_reviews.csv", index=False)

print("\n✅ Processed dataset saved successfully!")