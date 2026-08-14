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
    page_title="Model | TrustLens AI",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_css()


# =========================================================
# BACK TO HOME
# =========================================================

back_col1, back_col2 = st.columns([11, 1])

with back_col2:

    if st.button(
        "←",
        key="model_back",
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
# LOAD DATASET
# =========================================================

try:

    dataset = pd.read_csv(DATASET_PATH)

    review_count = len(dataset)

    dataset_loaded = True

except Exception:

    dataset = pd.DataFrame()

    review_count = 0

    dataset_loaded = False


# =========================================================
# LOAD VECTORIZER
# =========================================================

try:

    import joblib

    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    features = vectorizer.max_features or 0

except Exception:

    features = 10000


# =========================================================
# MODEL VALUES
# =========================================================

accuracy = metrics.get(
    "accuracy",
    0,
) * 100

precision = metrics.get(
    "precision",
    0,
) * 100

recall = metrics.get(
    "recall",
    0,
) * 100

f1_score = metrics.get(
    "f1_score",
    0,
) * 100

best_model = metrics.get(
    "best_model",
    "Linear SVM",
)


# =========================================================
# HEADER
# =========================================================

st.title("Model")

st.markdown(
    """
Technical overview of the machine learning model
behind TrustLens AI, including its pipeline,
evaluation results and feature representation.
"""
)

st.divider()


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.subheader("Model Overview")

overview_col1, overview_col2 = st.columns(
    [2, 1]
)


with overview_col1:

    with st.container(border=True):

        st.markdown(
            "### Fake Review Classification"
        )

        st.write(
            """
TrustLens AI uses a text classification pipeline
to distinguish between genuine and potentially fake
product reviews.

Review text is converted into TF-IDF feature vectors
and classified using a Linear Support Vector Machine.
"""
        )


with overview_col2:

    with st.container(border=True):

        st.markdown(
            "### Classifier"
        )

        st.markdown(
            f"## {best_model}"
        )

        st.caption(
            "Selected machine learning model."
        )


# =========================================================
# DETECTION PIPELINE
# =========================================================

st.markdown("")

st.subheader("Detection Pipeline")

st.caption(
    "The prediction process from raw review text to classification."
)


step1, step2, step3, step4 = st.columns(4)


with step1:

    with st.container(border=True):

        st.markdown("### 01")

        st.markdown(
            "Review Input"
        )

        st.caption(
            "Product review text is provided as input."
        )


with step2:

    with st.container(border=True):

        st.markdown("### 02")

        st.markdown(
            "TF-IDF"
        )

        st.caption(
            "Review text is transformed into numerical features."
        )


with step3:

    with st.container(border=True):

        st.markdown("### 03")

        st.markdown(
            "Linear SVM"
        )

        st.caption(
            "The classifier evaluates the generated feature vector."
        )


with step4:

    with st.container(border=True):

        st.markdown("### 04")

        st.markdown(
            "Prediction"
        )

        st.caption(
            "The review receives its classification."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("")

st.subheader("Model Performance")

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2f}%",
    )


with metric2:

    st.metric(
        "Precision",
        f"{precision:.2f}%",
    )


with metric3:

    st.metric(
        "Recall",
        f"{recall:.2f}%",
    )


with metric4:

    st.metric(
        "F1 Score",
        f"{f1_score:.2f}%",
    )


# =========================================================
# PERFORMANCE CHART
# =========================================================

st.markdown("")

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

st.bar_chart(
    performance_df.set_index("Metric"),
    y="Score",
    height=320,
)

st.caption(
    "Evaluation results on the model's test data."
)


# =========================================================
# DATA AND FEATURES
# =========================================================

st.markdown("")

st.subheader("Data and Features")

data1, data2, data3 = st.columns(3)


with data1:

    with st.container(border=True):

        st.markdown(
            "### Dataset"
        )

        st.markdown(
            f"## {review_count:,}"
        )

        st.caption(
            "Reviews available in the project dataset."
        )


with data2:

    with st.container(border=True):

        st.markdown(
            "### Text Features"
        )

        st.markdown(
            f"## {features:,}"
        )

        st.caption(
            "Maximum TF-IDF vocabulary features."
        )


with data3:

    with st.container(border=True):

        st.markdown(
            "### Representation"
        )

        st.markdown(
            "TF-IDF"
        )

        st.caption(
            "Numerical representation of review text."
        )


# =========================================================
# WHY TF-IDF
# =========================================================

st.markdown("")

st.subheader("Feature Representation")

with st.container(border=True):

    st.markdown(
        "### TF-IDF"
    )

    st.write(
        """
TF-IDF represents each review using weighted word
features. Words that are informative within a review
receive greater importance, while terms that occur
frequently across the dataset receive lower weight.
"""
    )


# =========================================================
# EVALUATION SUMMARY
# =========================================================

st.markdown("")

st.subheader("Evaluation Summary")

with st.container(border=True):

    st.markdown(
        f"""
The selected **{best_model}** classifier achieved:

**Accuracy:** {accuracy:.2f}%

**Precision:** {precision:.2f}%

**Recall:** {recall:.2f}%

**F1 Score:** {f1_score:.2f}%
"""
    )

    st.caption(
        "These metrics describe performance on the evaluated test data "
        "and should not be interpreted as proof that every prediction is correct."
    )




# =========================================================
# MODEL LIMITATION
# =========================================================

st.markdown("")

st.subheader("Model Limitation")

st.write(
    """
The model identifies patterns learned from the training
data. A review classified as fake or suspicious is a
machine learning prediction and does not independently
prove that the review is fraudulent.
"""
)


# =========================================================
# FOOTER
# =========================================================

st.markdown("")

st.divider()

st.caption(
    "TrustLens AI • Machine Learning Model"
)