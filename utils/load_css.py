import os
import streamlit as st


CSS_FILES = [
    "assets/css/globals.css",
    "assets/css/components.css",
]


def load_css():

    css = ""

    for file in CSS_FILES:

        if os.path.exists(file):

            with open(
                file,
                "r",
                encoding="utf-8",
            ) as f:

                css += f.read()

    st.markdown(
        f"<style>{css}</style>",
        unsafe_allow_html=True,
    )