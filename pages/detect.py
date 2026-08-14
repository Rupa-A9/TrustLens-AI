import time
import streamlit as st


from services.predictor import predict_review
from utils.load_css import load_css


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Detect Review - TrustLens AI",
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
# SAMPLE REVIEWS
# =========================================================



# =========================================================
# SESSION STATE
# =========================================================

if "detect_review_text" not in st.session_state:
    st.session_state.detect_review_text = ""

if "last_detection" not in st.session_state:
    st.session_state.last_detection = None


# =========================================================
# HEADER
# =========================================================

st.title("Detect Review")

st.markdown(
    """
Analyze a product review using the trained
**TF-IDF + Linear SVM** model.
"""
)

st.divider()


# =========================================================
# SAMPLE REVIEW BUTTONS
# =========================================================




# =========================================================
# REVIEW INPUT
# =========================================================

st.subheader("Product Review")

review_text = st.text_area(
    "Review",
    value=st.session_state.detect_review_text,
    height=220,
    placeholder=(
        "Paste or type a product review here..."
    ),
    label_visibility="collapsed",
)

st.session_state.detect_review_text = review_text


# =========================================================
# LIVE REVIEW STATISTICS
# =========================================================

characters = len(review_text)

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


stat_col1, stat_col2, stat_col3 = st.columns(3)

with stat_col1:
    st.metric(
        "Characters",
        f"{characters:,}",
    )

with stat_col2:
    st.metric(
        "Words",
        f"{words:,}",
    )

with stat_col3:
    st.metric(
        "Lines",
        f"{lines:,}",
    )


st.markdown("")


# =========================================================
# ANALYZE
# =========================================================

if st.button(
    "Analyze Review",
    type="primary",
    use_container_width=True,
):

    if not review_text.strip():

        st.warning(
            "Please enter a product review before analyzing."
        )

    else:

        start_time = time.perf_counter()

        try:

            with st.spinner(
                "Analyzing review with Linear SVM..."
            ):

                prediction_result = predict_review(
                    review_text
                )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

            st.stop()

        processing_time = (
            time.perf_counter() - start_time
        )


        # =================================================
        # READ PREDICTION RESULT
        # =================================================

        if isinstance(
            prediction_result,
            dict,
        ):

            label = str(
                prediction_result.get(
                    "label",
                    "Unknown",
                )
            )

            confidence = float(
                prediction_result.get(
                    "confidence",
                    0,
                )
            )

            prediction = prediction_result.get(
                "prediction"
            )

        elif isinstance(
            prediction_result,
            (tuple, list),
        ):

            label = str(
                prediction_result[0]
            )

            confidence = float(
                prediction_result[1]
            ) if len(prediction_result) > 1 else 0

            prediction = None

        else:

            label = str(
                prediction_result
            )

            confidence = 0

            prediction = None


        # =================================================
        # CONFIDENCE NORMALIZATION
        # =================================================

        if confidence > 1:
            confidence = confidence / 100

        confidence = max(
            0.0,
            min(
                confidence,
                1.0,
            ),
        )

        confidence_percentage = (
            confidence * 100
        )


        # =================================================
        # RESULT TYPE
        # =================================================

        label_lower = label.lower()

        is_genuine = any(
            word in label_lower
            for word in [
                "genuine",
                "real",
                "authentic",
            ]
        )


        if is_genuine:

            result_title = "Genuine Review"

            result_message = (
                "The model classified this review "
                "as likely genuine."
            )

        else:

            result_title = (
                "Fake / Suspicious Review"
            )

            result_message = (
                "The model classified this review "
                "as potentially fake or suspicious."
            )


        # =================================================
        # SAVE RESULT
        # =================================================

        st.session_state.last_detection = {
            "review": review_text,
            "label": result_title,
            "message": result_message,
            "confidence": confidence_percentage,
            "processing_time": processing_time,
            "characters": characters,
            "words": words,
            "lines": lines,
            "prediction": prediction,
        }


# =========================================================
# DISPLAY RESULT
# =========================================================

result = st.session_state.last_detection


if result is not None:

    st.divider()

    st.subheader(" Detection Result")


    # =====================================================
    # RESULT STATUS
    # =====================================================

    if result["label"] == "Genuine Review":

        st.success(
            f"✅ {result['label']}"
        )

    else:

        st.warning(
            f"⚠️ {result['label']}"
        )


    st.info(
        result["message"]
    )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    st.markdown(
        "### Prediction Confidence"
    )

    confidence = result["confidence"]

    st.progress(
        confidence / 100
    )

    confidence_col1, confidence_col2 = st.columns(
        [4, 1]
    )

    with confidence_col1:

        st.caption(
            "Model confidence"
        )

    with confidence_col2:

        st.markdown(
            f"### {confidence:.2f}%"
        )


    # =====================================================
    # REVIEW STATISTICS
    # =====================================================

    st.markdown(
        "###  Review Statistics"
    )

    result_col1, result_col2, result_col3, result_col4 = (
        st.columns(4)
    )


    with result_col1:

        st.metric(
            "Characters",
            f"{result['characters']:,}",
        )


    with result_col2:

        st.metric(
            "Words",
            f"{result['words']:,}",
        )


    with result_col3:

        st.metric(
            "Lines",
            f"{result['lines']:,}",
        )


    with result_col4:

        st.metric(
            "Processing Time",
            f"{result['processing_time']:.3f}s",
        )


    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.markdown(
        "###  Model Information"
    )

    model_col1, model_col2, model_col3 = st.columns(3)


    with model_col1:

        with st.container(border=True):

            st.markdown(
                "###  TF-IDF"
            )

            st.caption(
                "Text vectorization converts "
                "review text into numerical features."
            )


    with model_col2:

        with st.container(border=True):

            st.markdown(
                "###  Linear SVM"
            )

            st.caption(
                "The trained classifier predicts "
                "the review category."
            )


    with model_col3:

        with st.container(border=True):

            st.markdown(
                "### 🛡️ TrustLens AI"
            )

            st.caption(
                "AI-powered fake review detection."
            )


    # =====================================================
    # REVIEW THAT WAS ANALYZED
    # =====================================================

    st.markdown(
        "###  Analyzed Review"
    )

    with st.container(border=True):

        st.write(
            result["review"]
        )


    # =====================================================
    # PDF REPORT
    # =====================================================

    st.markdown(
        "###  Download Report"
    )

    try:

        from utils.pdf import create_report

        pdf_data = create_report(
            review=result["review"],
            label=result["label"],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
        )

        st.download_button(
            label="⬇️ Download PDF Report",
            data=pdf_data,
            file_name="trustlens_review_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except ImportError:

        st.warning(
            "PDF support is not installed."
        )

        st.code(
            "pip install reportlab"
        )

    except Exception as error:

        st.error(
            f"Could not generate PDF report: {error}"
        )


