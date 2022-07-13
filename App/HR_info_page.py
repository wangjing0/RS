import streamlit as st
import pickle
import numpy as np
from PIL import Image

def HR_info_page():
    width = 600
    st.title("...about heart health...")
    #st.header("")
    image = Image.open('images/US_restingHR.webp')
    st.image(image, caption=" ", width=width)
    st.markdown('#')
    image = Image.open('images/countries_restingHR.png')
    st.image(image, caption=" ", width=width)
    st.markdown('#')
    #st.header("Resting Heart Rate(bpm)")
    image = Image.open('images/restingHR.jpeg')
    st.image(image, caption="Resting Heart Rate (bpm) ", width=width)
    st.markdown('#')
    image = Image.open('images/restingHR_age_gender.webp')
    st.image(image, caption=" ", width=width)
    st.markdown('#')
    image = Image.open('images/restingHR_activity.webp')
    st.image(image, caption=" ", width=width)
    st.markdown('#')
    image = Image.open('images/HR_exercise.png')
    st.image(image, caption=" ", width=width)
    st.markdown('#')
    image = Image.open('images/Immediate-heart-rate-reponse.jpeg')
    st.image(image, caption=" ", width=width)
    st.markdown('#')
   
    
    
    