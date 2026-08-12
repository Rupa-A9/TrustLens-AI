import streamlit as st
import pandas as pd

from services.predictor import predict_review
from utils.load_css import load_css


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Batch Detection - TrustLens AI",
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
# HEADER
# =========================================================

st.title("Batch Detection")

st.markdown(
    """
Upload multiple product reviews, analyze them with
the trained **TF-IDF + Linear SVM** model, and download
the complete analysis results.
"""
)

st.divider()


# =========================================================
# UPLOAD CSV
# =========================================================

st.subheader("📤 Upload Reviews")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV containing your product reviews.",
)


# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            f"Unable to read the CSV file: {error}"
        )

        st.stop()


    if df.empty:

        st.warning(
            "The uploaded CSV is empty."
        )

        st.stop()


    # =====================================================
    # FILE INFORMATION
    # =====================================================

    file_col1, file_col2, file_col3 = st.columns(3)

    with file_col1:

        st.metric(
            "Reviews",
            f"{len(df):,}",
        )

    with file_col2:

        st.metric(
            "Columns",
            f"{len(df.columns):,}",
        )

    with file_col3:

        st.metric(
            "File Size",
            f"{uploaded_file.size / 1024:.1f} KB",
        )


    st.markdown("")


    # =====================================================
    # PREVIEW
    # =====================================================

    st.subheader("Review Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # FIND REVIEW COLUMN
    # =====================================================

    st.subheader("Review Column")

    columns = list(
        df.columns
    )

    possible_review_columns = [
        "review",
        "review_text",
        "reviewText",
        "text",
        "content",
        "review_body",
        "Review",
        "Review Text",
    ]

    detected_column = None

    for column in possible_review_columns:

        if column in columns:

            detected_column = column
            break


    # If no common name exists,
    # find the first text column.

    if detected_column is None:

        text_columns = list(
            df.select_dtypes(
                include=["object"]
            ).columns
        )

        if text_columns:

            detected_column = text_columns[0]


    if detected_column is None:

        st.error(
            "No review/text column was found."
        )

        st.info(
            "Please upload a CSV containing a text column."
        )

        st.stop()


    selected_column = st.selectbox(
        "Select the column containing review text",
        options=columns,
        index=columns.index(
            detected_column
        ),
    )


    st.caption(
        f"Selected review column: `{selected_column}`"
    )


    # =====================================================
    # ANALYZE BUTTON
    # =====================================================

    st.markdown("")

    if st.button(
        "Analyze All Reviews",
        type="primary",
        use_container_width=True,
    ):

        results = []

        total_reviews = len(df)

        progress_bar = st.progress(0)

        status = st.empty()


        # =================================================
        # ANALYZE EACH REVIEW
        # =================================================

        for index, review in enumerate(
            df[selected_column].fillna("")
        ):

            review = str(review).strip()

            status.write(
                f"Analyzing review "
                f"{index + 1:,} of "
                f"{total_reviews:,}..."
            )


            # ---------------------------------------------
            # EMPTY REVIEW
            # ---------------------------------------------

            if not review:

                results.append(
                    {
                        "Prediction": "Empty Review",
                        "Confidence (%)": 0.0,
                    }
                )

                progress_bar.progress(
                    (index + 1)
                    / total_reviews
                )

                continue


            # ---------------------------------------------
            # PREDICTION
            # ---------------------------------------------

            try:

                prediction_result = predict_review(
                    review
                )


                # -----------------------------------------
                # READ RESULT
                # -----------------------------------------

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

                    prediction_value = (
                        prediction_result.get(
                            "prediction",
                            "",
                        )
                    )


                elif isinstance(
                    prediction_result,
                    (tuple, list),
                ):

                    label = str(
                        prediction_result[0]
                    )

                    confidence = (
                        float(
                            prediction_result[1]
                        )
                        if len(
                            prediction_result
                        ) > 1
                        else 0
                    )

                    prediction_value = ""


                else:

                    label = str(
                        prediction_result
                    )

                    confidence = 0

                    prediction_value = ""


                # -----------------------------------------
                # NORMALIZE CONFIDENCE
                # -----------------------------------------

                if confidence > 1:

                    confidence = (
                        confidence / 100
                    )


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


                # -----------------------------------------
                # FINAL LABEL
                # -----------------------------------------

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

                    final_prediction = (
                        "Genuine Review"
                    )

                else:

                    final_prediction = (
                        "Fake / Suspicious Review"
                    )


                results.append(
                    {
                        "Prediction": final_prediction,
                        "Confidence (%)": round(
                            confidence_percentage,
                            2,
                        ),
                    }
                )


            except Exception as error:

                results.append(
                    {
                        "Prediction": "Error",
                        "Confidence (%)": 0.0,
                    }
                )


            # -----------------------------------------
            # PROGRESS
            # -----------------------------------------

            progress_bar.progress(
                (index + 1)
                / total_reviews
            )


        status.empty()

        progress_bar.empty()


        # =================================================
        # CREATE RESULT DATAFRAME
        # =================================================

        result_df = df.copy()

        prediction_df = pd.DataFrame(
            results
        )


        result_df[
            "TrustLens Prediction"
        ] = prediction_df[
            "Prediction"
        ].values


        result_df[
            "Confidence (%)"
        ] = prediction_df[
            "Confidence (%)"
        ].values


        # Save results
        st.session_state.batch_results = (
            result_df
        )


        st.success(
            f"Analysis completed successfully for "
            f"{total_reviews:,} reviews."
        )


# =========================================================
# SHOW ANALYSIS RESULTS
# =========================================================

if "batch_results" in st.session_state:

    result_df = (
        st.session_state.batch_results
    )


    st.divider()

    st.subheader(
        "Analysis Results"
    )


    # =====================================================
    # SUMMARY
    # =====================================================

    total = len(
        result_df
    )

    genuine = len(
        result_df[
            result_df[
                "TrustLens Prediction"
            ]
            == "Genuine Review"
        ]
    )

    fake = len(
        result_df[
            result_df[
                "TrustLens Prediction"
            ]
            == "Fake / Suspicious Review"
        ]
    )

    errors = len(
        result_df[
            result_df[
                "TrustLens Prediction"
            ]
            == "Error"
        ]
    )


    genuine_percentage = (
        genuine / total * 100
        if total
        else 0
    )

    fake_percentage = (
        fake / total * 100
        if total
        else 0
    )


    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "📄 Total Reviews",
            f"{total:,}",
        )

    with c2:

        st.metric(
            "✅ Genuine",
            f"{genuine:,}",
            f"{genuine_percentage:.1f}%",
        )

    with c3:

        st.metric(
            "⚠️ Fake / Suspicious",
            f"{fake:,}",
            f"{fake_percentage:.1f}%",
        )

    with c4:

        st.metric(
            "❌ Errors",
            f"{errors:,}",
        )


    # =====================================================
    # RESULTS TABLE
    # =====================================================

    st.markdown(
        "### Detailed Analysis"
    )

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # FILTER
    # =====================================================

    st.markdown(
        "###  Filter Results"
    )

    filter_option = st.selectbox(
        "Show",
        [
            "All Reviews",
            "Genuine Reviews",
            "Fake / Suspicious Reviews",
            "Errors",
        ],
    )


    if filter_option == "Genuine Reviews":

        filtered_df = result_df[
            result_df[
                "TrustLens Prediction"
            ]
            == "Genuine Review"
        ]


    elif filter_option == "Fake / Suspicious Reviews":

        filtered_df = result_df[
            result_df[
                "TrustLens Prediction"
            ]
            == "Fake / Suspicious Review"
        ]


    elif filter_option == "Errors":

        filtered_df = result_df[
            result_df[
                "TrustLens Prediction"
            ]
            == "Error"
        ]


    else:

        filtered_df = result_df


    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )


    # =====================================================
    # DOWNLOAD ANALYZED RESULTS
    # =====================================================

    st.markdown(
        "### 📥 Download Analysis"
    )

    st.caption(
        "Download the original reviews together with "
        "the TrustLens AI predictions and confidence scores."
    )


    analyzed_csv = result_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Analyzed Results",
        data=analyzed_csv,
        file_name="trustlens_analyzed_reviews.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# NO FILE
# =========================================================

elif uploaded_file is None:

    st.info(
        "Upload a CSV file above to start batch detection."
    )


