import streamlit as st
from supabase import create_client


def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


def signup(email, password):

    supabase = get_supabase()

    response = supabase.auth.sign_up({
        "email": email,
        "password": password
    })

    return response


def login(email, password):

    supabase = get_supabase()

    response = supabase.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    return response


def logout():

    supabase = get_supabase()

    supabase.auth.sign_out()

    st.session_state.logged_in = False
    st.session_state.user = None

    st.rerun()


def require_login():

    if not st.session_state.get("logged_in", False):

        st.warning(
            "Please log in to access this page."
        )

        if st.button(
            "Go to Login",
            type="primary"
        ):
            st.switch_page("pages/login.py")

        st.stop()