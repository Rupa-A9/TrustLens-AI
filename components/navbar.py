import os
import streamlit as st


def navbar():

    html_file = os.path.join(
        os.path.dirname(__file__),
        "navbar.html"
    )

    with open(
        html_file,
        "r",
        encoding="utf-8"
    ) as file:
        html = file.read()

    st.html(html)