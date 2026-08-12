from pathlib import Path

# =========================================================
# PROJECT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

APP_NAME = "TrustLens AI"
TAGLINE = "AI Powered Fake Review Detection"
VERSION = "1.0.0"

AUTHOR = "Siripurapu Rupasri"
GITHUB_URL = "https://github.com/Rupa-A9"

# =========================================================
# PATHS
# =========================================================

ASSETS_DIR = BASE_DIR / "assets"
CSS_DIR = ASSETS_DIR / "css"

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "processed_reviews.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"

# =========================================================
# PROJECT INFORMATION
# =========================================================

DATASET_NAME = "Amazon Product Reviews"

TFIDF_FEATURES = 10000

BEST_MODEL = "Linear SVM"

# =========================================================
# QUICK ACTIONS
# =========================================================

QUICK_ACTIONS = [
    {
        "title": "Detect Review",
        "description": "Analyze a single product review.",
        "icon": "🔍",
        "page": "pages/detect.py",
    },
    {
        "title": "Batch Detection",
        "description": "Analyze multiple reviews from a CSV file.",
        "icon": "📂",
        "page": "pages/batch.py",
    },
    {
        "title": "Dashboard",
        "description": "View model performance and project metrics.",
        "icon": "📊",
        "page": "pages/dashboard.py",
    },
    {
        "title": "Model Insights",
        "description": "Understand TF-IDF and Linear SVM.",
        "icon": "🧠",
        "page": "pages/insights.py",
    },
]