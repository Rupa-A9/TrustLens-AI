import streamlit as st

from pages.home import show_home


st.set_page_config(
    page_title="TrustLens AI",
    page_icon="🛡️",
    layout="wide",
)


page = st.query_params.get("page", "home")


if page == "detect":
    st.switch_page("pages/detect.py")

elif page == "batch":
    st.switch_page("pages/batch.py")

elif page == "dashboard":
    st.switch_page("pages/dashboard.py")


else:
    show_home()