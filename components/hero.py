import os
import streamlit as st


def hero():

    html_file = os.path.join(
        os.path.dirname(__file__),
        "hero.html",
    )

    with open(
        html_file,
        "r",
        encoding="utf-8",
    ) as file:
        html = file.read()

    st.html(html)