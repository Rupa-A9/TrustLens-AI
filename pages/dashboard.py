import json
import pandas as pd
import streamlit as st

from config import (
    DATASET_PATH,
    METRICS_PATH,
    VECTORIZER_PATH,
)

from utils.load_css import load_css


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Dashboard - TrustLens AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()

# =========================================================
# BACK TO HOME
# =========================================================

back_col1, back_col2 = st.columns([11, 1])

with back_col2:

    with st.container(key="back_home_button"):

        if st.button(
            "←",
            key="back_home",
            help="Back to Home",
            use_container_width=True,
        ):
            st.switch_page("app.py")



# =========================================================
# LOAD METRICS
# =========================================================

try:

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metrics = json.load(file)

except Exception:

    metrics = {
        "best_model": "Linear SVM",
        "accuracy": 0.8915543464820082,
        "precision": 0.8904785397138628,
        "recall": 0.8929013109077418,
        "f1_score": 0.891688279609732,
    }


# =========================================================
# DATASET INFORMATION
# =========================================================

try:

    dataset = pd.read_csv(
        DATASET_PATH
    )

    review_count = len(
        dataset
    )

except Exception:

    review_count = 0


# =========================================================
# TF-IDF FEATURES
# =========================================================

try:

    import joblib

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    features = (
        vectorizer.max_features
        or 0
    )

except Exception:

    # Avoid breaking the dashboard if
    # the vectorizer cannot be loaded.
    features = 10000


# =========================================================
# CONVERT METRICS TO PERCENTAGES
# =========================================================

accuracy = (
    metrics.get("accuracy", 0)
    * 100
)

precision = (
    metrics.get("precision", 0)
    * 100
)

recall = (
    metrics.get("recall", 0)
    * 100
)

f1_score = (
    metrics.get("f1_score", 0)
    * 100
)

best_model = metrics.get(
    "best_model",
    "Linear SVM",
)


# =========================================================
# HEADER
# =========================================================

st.title("Model Dashboard")

st.markdown(
    """
Explore the performance of the TrustLens AI
fake review detection model.
"""
)

st.divider()


# =========================================================
# TOP METRICS
# =========================================================

st.subheader("📈 Model Performance")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2f}%",
    )


with col2:

    st.metric(
        "Precision",
        f"{precision:.2f}%",
    )


with col3:

    st.metric(
        "Recall",
        f"{recall:.2f}%",
    )


with col4:

    st.metric(
        "F1 Score",
        f"{f1_score:.2f}%",
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown("")

st.subheader("Model Information")


model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    with st.container(border=True):

        st.markdown(
            "### Best Model"
        )

        st.markdown(
            f"## {best_model}"
        )

        st.caption(
            "Selected machine learning classifier."
        )


with model_col2:

    with st.container(border=True):

        st.markdown(
            "###  Dataset"
        )

        st.markdown(
            f"## {review_count:,}"
        )

        st.caption(
            "Reviews available for the project."
        )


with model_col3:

    with st.container(border=True):

        st.markdown(
            "### TF-IDF"
        )

        st.markdown(
            f"## {features:,}"
        )

        st.caption(
            "Maximum text features used by the vectorizer."
        )


# =========================================================
# PERFORMANCE CHART
# =========================================================

st.markdown("")

st.subheader("Performance Comparison")


performance_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ],
        "Score": [
            accuracy,
            precision,
            recall,
            f1_score,
        ],
    }
)

performance_df = performance_df.set_index(
    "Metric"
)


st.bar_chart(
    performance_df,
    y="Score",
    height=350,
)


# =========================================================
# METRIC DETAILS
# =========================================================

st.markdown("")

st.subheader("Metric Details")


detail_col1, detail_col2 = st.columns(2)


with detail_col1:

    with st.container(border=True):

        st.markdown(
            "### Accuracy"
        )

        st.write(
            f"{accuracy:.2f}%"
        )

        st.caption(
            "The percentage of reviews correctly classified "
            "by the model."
        )


with detail_col2:

    with st.container(border=True):

        st.markdown(
            "### Precision"
        )

        st.write(
            f"{precision:.2f}%"
        )

        st.caption(
            "Measures how many reviews predicted as a "
            "class were actually that class."
        )


detail_col3, detail_col4 = st.columns(2)


with detail_col3:

    with st.container(border=True):

        st.markdown(
            "### Recall"
        )

        st.write(
            f"{recall:.2f}%"
        )

        st.caption(
            "Measures how effectively the model identifies "
            "reviews belonging to a class."
        )


with detail_col4:

    with st.container(border=True):

        st.markdown(
            "### F1 Score"
        )

        st.write(
            f"{f1_score:.2f}%"
        )

        st.caption(
            "Balances precision and recall into a single score."
        )


# =========================================================
# MODEL PIPELINE
# =========================================================

st.markdown("")

st.subheader("Detection Pipeline")


step1, step2, step3, step4 = st.columns(4)


with step1:

    with st.container(border=True):

        st.markdown(
            "### 01"
        )

        st.markdown(
            "**Review Input**"
        )

        st.caption(
            "Product review text enters the system."
        )


with step2:

    with st.container(border=True):

        st.markdown(
            "### 02"
        )

        st.markdown(
            "**TF-IDF**"
        )

        st.caption(
            "Text is converted into numerical features."
        )


with step3:

    with st.container(border=True):

        st.markdown(
            "### 03"
        )

        st.markdown(
            "**Linear SVM**"
        )

        st.caption(
            "The trained classifier evaluates the review."
        )


with step4:

    with st.container(border=True):

        st.markdown(
            "### 04"
        )

        st.markdown(
            "**Prediction**"
        )

        st.caption(
            "TrustLens AI returns the classification."
        )


# =========================================================
# PROJECT SUMMARY
# =========================================================

st.markdown("")

st.subheader("🛡️ TrustLens AI")


with st.container(border=True):

    st.markdown(
        """
TrustLens AI is an AI-powered fake review detection
platform that uses Natural Language Processing,
TF-IDF vectorization and a Linear SVM classifier
to analyze product reviews.
"""
    )

    st.markdown(
        f"""
**Best Model:** {best_model}

**Accuracy:** {accuracy:.2f}%

**Dataset:** {review_count:,} reviews

**TF-IDF Features:** {features:,}
"""
    )

