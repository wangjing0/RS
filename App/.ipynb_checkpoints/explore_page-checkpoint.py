import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
import geopandas as gpd
from geopandas import GeoDataFrame
from PIL import Image

@st.cache
def load_data():
    df = pd.read_pickle("./data/df_proper_cleaned.pkl")
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Activities")

    st.write(
    """
    ### Geographic distribution of our users
    """
    )
    
    image = Image.open('./images/geo_density.png')
    st.image(image, caption="Users Over the world ",use_column_width=True)