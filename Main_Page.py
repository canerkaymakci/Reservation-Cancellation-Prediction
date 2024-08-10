import streamlit as st

st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 4em;
        font-weight: bold;
    }
    .centered-title-2 {
        text-align: center;
        font-size: 3em;
    }
    </style>
    <div class="centered-title">
        Miuuloria Resort
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <div class="centered-title-2">
        Reception System
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("Contact")
st.sidebar.write("Rabia Nur Gülmez")
st.sidebar.write("Caner Kaymakçı")
