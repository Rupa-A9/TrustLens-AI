import time
from pathlib import Path

import joblib

from config import MODEL_PATH, VECTORIZER_PATH


# =========================================================
# MODEL PATHS
# =========================================================

MODEL_FILE = Path(MODEL_PATH)
VECTORIZER_FILE = Path(VECTORIZER_PATH)


# =========================================================
# LOAD MODEL
# =========================================================

_model = None
_vectorizer = None


def load_model():

    global _model
    global _vectorizer

    if _model is None:

        if not MODEL_FILE.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_FILE}"
            )

        _model = joblib.load(
            MODEL_FILE
        )

    if _vectorizer is None:

        if not VECTORIZER_FILE.exists():
            raise FileNotFoundError(
                f"Vectorizer file not found: {VECTORIZER_FILE}"
            )

        _vectorizer = joblib.load(
            VECTORIZER_FILE
        )

    return _model, _vectorizer


# =========================================================
# PREDICT REVIEW
# =========================================================

def predict_review(review_text):

    if not isinstance(
        review_text,
        str,
    ):
        raise TypeError(
            "Review must be a string."
        )

    review_text = review_text.strip()

    if not review_text:

        raise ValueError(
            "Review cannot be empty."
        )

    # Load trained model and vectorizer
    model, vectorizer = load_model()

    # Convert review into TF-IDF features
    features = vectorizer.transform(
        [review_text]
    )

    # Make prediction
    prediction = model.predict(
        features
    )[0]

    # -----------------------------------------------------
    # Prediction label
    # -----------------------------------------------------

    if str(prediction) == "1":

        label = "Genuine Review"

    else:

        label = "Fake / Suspicious Review"


    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = 0.0

    try:

        if hasattr(
            model,
            "decision_function",
        ):

            decision = model.decision_function(
                features
            )

            score = float(
                abs(decision[0])
            )

            # Convert SVM decision score into
            # a simple 0-100 representation.
            confidence = (
                score
                / (1.0 + score)
                * 100
            )

    except Exception:

        confidence = 0.0


    # Keep confidence between 0 and 100
    confidence = max(
        0.0,
        min(
            confidence,
            100.0,
        ),
    )


    return {
        "prediction": prediction,
        "label": label,
        "confidence": confidence,
    }


# =========================================================
# REVIEW STATISTICS
# =========================================================

def review_statistics(review_text):

    if not isinstance(
        review_text,
        str,
    ):
        review_text = str(
            review_text
        )

    start_time = time.perf_counter()

    characters = len(
        review_text
    )

    words = (
        len(review_text.split())
        if review_text.strip()
        else 0
    )

    lines = (
        len(review_text.splitlines())
        if review_text.strip()
        else 0
    )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    return {
        "characters": characters,
        "words": words,
        "lines": lines,
        "processing_time": (
            f"{processing_time:.4f}s"
        ),
    }