import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('./data/saved_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

regressor = model["model"]
le_country = model["le_country"]
le_education =model["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)
        X = pd.DataFrame(data=X, columns=['country','education'])

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")