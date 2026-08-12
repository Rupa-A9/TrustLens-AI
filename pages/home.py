import streamlit as st

from utils.load_css import load_css
from components.navbar import navbar
from components.hero import hero


def show_home():

    load_css()

    navbar()

    hero()