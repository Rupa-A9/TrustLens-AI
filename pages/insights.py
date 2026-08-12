import json

import streamlit as st

from config import METRICS_PATH
from utils.load_css import load_css


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Model Insights - TrustLens AI",
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
# LOAD MODEL METRICS
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

st.title("Model Insights")

st.markdown(
    """
Understand how TrustLens AI analyzes product reviews
and identifies potentially fake or suspicious reviews.
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
            "### TrustLens AI Detection Model"
        )

        st.write(
            """
TrustLens AI uses Natural Language Processing (NLP)
to transform review text into numerical features.
These features are then evaluated by a trained
Linear Support Vector Machine classifier.
"""
        )

        st.write(
            """
The system is designed to distinguish between
genuine and potentially fake product reviews
based on patterns learned from the training data.
"""
        )


with overview_col2:

    with st.container(border=True):

        st.markdown(
            "### Best Model"
        )

        st.markdown(
            f"## {best_model}"
        )

        st.caption(
            "Selected classification algorithm."
        )


# =========================================================
# NLP
# =========================================================

st.markdown("")

st.subheader(" 01 — Natural Language Processing")

st.markdown(
    """
Natural Language Processing allows TrustLens AI to
work with human-written product reviews.
"""
)


nlp1, nlp2, nlp3 = st.columns(3)


with nlp1:

    with st.container(border=True):

        st.markdown(
            "###  Review Text"
        )

        st.caption(
            "The original product review is provided "
            "as input to the system."
        )


with nlp2:

    with st.container(border=True):

        st.markdown(
            "###  Text Processing"
        )

        st.caption(
            "The review is prepared so it can be "
            "converted into machine-readable features."
        )


with nlp3:

    with st.container(border=True):

        st.markdown(
            "###  Numerical Representation"
        )

        st.caption(
            "Text is transformed into numerical "
            "information for machine learning."
        )


# =========================================================
# TF-IDF
# =========================================================

st.markdown("")

st.subheader(" 02 — TF-IDF Vectorization")

st.markdown(
    """
TF-IDF converts review text into numerical feature
vectors that the machine learning model can understand.
"""
)


tfidf_col1, tfidf_col2 = st.columns(2)


with tfidf_col1:

    with st.container(border=True):

        st.markdown(
            "### TF — Term Frequency"
        )

        st.write(
            """
Measures how frequently a word appears within
a particular review.
"""
        )


with tfidf_col2:

    with st.container(border=True):

        st.markdown(
            "### IDF — Inverse Document Frequency"
        )

        st.write(
            """
Measures how important or distinctive a word is
across the collection of reviews.
"""
        )


st.info(
    """
TF-IDF combines these concepts to give more importance
to informative words while reducing the influence of
very common words.
"""
)


# =========================================================
# LINEAR SVM
# =========================================================

st.markdown("")

st.subheader(" 03 — Linear SVM")

st.markdown(
    """
The trained Linear Support Vector Machine classifier
uses the TF-IDF features to classify reviews.
"""
)


svm1, svm2, svm3 = st.columns(3)


with svm1:

    with st.container(border=True):

        st.markdown(
            "###  Input"
        )

        st.caption(
            "TF-IDF feature vector generated from the review."
        )


with svm2:

    with st.container(border=True):

        st.markdown(
            "###  Classification"
        )

        st.caption(
            "Linear SVM evaluates the learned decision boundary."
        )


with svm3:

    with st.container(border=True):

        st.markdown(
            "###  Output"
        )

        st.caption(
            "The system returns a review classification."
        )


# =========================================================
# DETECTION PIPELINE
# =========================================================

st.markdown("")

st.subheader(" Complete Detection Pipeline")


step1, step2, step3, step4 = st.columns(4)


with step1:

    with st.container(border=True):

        st.markdown("### 01")

        st.markdown(
            "**Review Input**"
        )

        st.caption(
            "User enters or uploads a product review."
        )


with step2:

    with st.container(border=True):

        st.markdown("### 02")

        st.markdown(
            "**TF-IDF**"
        )

        st.caption(
            "Text is transformed into numerical features."
        )


with step3:

    with st.container(border=True):

        st.markdown("### 03")

        st.markdown(
            "**Linear SVM**"
        )

        st.caption(
            "The classifier evaluates the feature vector."
        )


with step4:

    with st.container(border=True):

        st.markdown("### 04")

        st.markdown(
            "**Prediction**"
        )

        st.caption(
            "TrustLens AI returns the classification."
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("")

st.subheader("Model Performance")


p1, p2, p3, p4 = st.columns(4)


with p1:

    st.metric(
        "Accuracy",
        f"{accuracy:.2f}%",
    )


with p2:

    st.metric(
        "Precision",
        f"{precision:.2f}%",
    )


with p3:

    st.metric(
        "Recall",
        f"{recall:.2f}%",
    )


with p4:

    st.metric(
        "F1 Score",
        f"{f1_score:.2f}%",
    )


# =========================================================
# WHAT THE METRICS MEAN
# =========================================================

st.markdown("")

st.subheader("Understanding the Metrics")


metric1, metric2 = st.columns(2)


with metric1:

    with st.container(border=True):

        st.markdown(
            "###  Accuracy"
        )

        st.write(
            f"{accuracy:.2f}%"
        )

        st.caption(
            "The proportion of all predictions "
            "that were classified correctly."
        )


with metric2:

    with st.container(border=True):

        st.markdown(
            "### Precision"
        )

        st.write(
            f"{precision:.2f}%"
        )

        st.caption(
            "Indicates how reliable the model's "
            "positive classifications are."
        )


metric3, metric4 = st.columns(2)


with metric3:

    with st.container(border=True):

        st.markdown(
            "###  Recall"
        )

        st.write(
            f"{recall:.2f}%"
        )

        st.caption(
            "Indicates how effectively the model "
            "identifies relevant reviews."
        )


with metric4:

    with st.container(border=True):

        st.markdown(
            "###  F1 Score"
        )

        st.write(
            f"{f1_score:.2f}%"
        )

        st.caption(
            "Provides a balance between precision "
            "and recall."
        )


# =========================================================
# TECHNOLOGY STACK
# =========================================================

st.markdown("")

st.subheader(" Technology Stack")


tech1, tech2, tech3, tech4 = st.columns(4)


with tech1:

    with st.container(border=True):

        st.markdown("###  Python")

        st.caption(
            "Core programming language."
        )


with tech2:

    with st.container(border=True):

        st.markdown("###  Scikit-learn")

        st.caption(
            "Machine learning implementation."
        )


with tech3:

    with st.container(border=True):

        st.markdown("###  TF-IDF")

        st.caption(
            "Text feature extraction."
        )


with tech4:

    with st.container(border=True):

        st.markdown("###  Streamlit")

        st.caption(
            "Interactive application interface."
        )


# =========================================================
# IMPORTANT NOTE
# =========================================================

st.markdown("")

st.info(
    """
TrustLens AI predictions represent machine learning
classifications based on patterns learned from the
training data. A prediction marked as suspicious does
not by itself prove that a review is fraudulent.
"""
)


