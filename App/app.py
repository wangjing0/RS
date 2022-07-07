import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page
from HR_info_page import HR_info_page


page = st.sidebar.selectbox("I Want To", ("Ask", "Explore", "Known some health trivia"))

def navigation():
    try:
        path = st.experimental_get_query_params()['p'][0]
    except Exception as e:
        st.error('Please use the main app.')
        return None
    return path

if page == "Explore":
    show_explore_page()
elif page == "Ask":
    show_predict_page()
else:
    HR_info_page()