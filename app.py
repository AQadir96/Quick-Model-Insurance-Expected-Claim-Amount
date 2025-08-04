import streamlit as st
import pandas as pd
from joblib import load

# Load trained model
model = load('claim_model.joblib')

# App title and headers
st.title('Insurance Claim Prediction App')
st.subheader('Predict expected insurance claim amount based on customer profile.')

# --- Sidebar Inputs ---
st.sidebar.header("Customer Profile Input")

# Categorical Inputs
gender = st.sidebar.selectbox("Gender", ['male', 'female'])
diabetic = st.sidebar.selectbox("Diabetic", ['yes', 'no'])
smoker = st.sidebar.selectbox("Smoker", ['yes', 'no'])
region = st.sidebar.selectbox("Region", ['southeast', 'southwest', 'northeast', 'northwest'])

# Numerical Inputs
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=35)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, value=25.0)
bloodpressure = st.sidebar.slider("Blood Pressure", min_value=60, max_value=180, value=120)
children = st.sidebar.slider("Number of Children", min_value=0, max_value=5, value=1)

# --- Combine inputs into DataFrame ---
input_data = pd.DataFrame({
    'age': [age],
    'gender': [gender],
    'bmi': [bmi],
    'bloodpressure': [bloodpressure],
    'diabetic': [diabetic],
    'children': [children],
    'smoker': [smoker],
    'region': [region]
})

# --- Prediction Button ---
if st.button("Predict Claim Amount"):
    predicted_claim = model.predict(input_data)[0]
    st.markdown(f"## Predicted Claim: **${predicted_claim:,.2f}**")

    # Risk assessment
    threshold = 15000
    if predicted_claim >= threshold:
        st.warning("High Cost Customer: Consider higher premiums or risk mitigation.")
    else:
        st.success("Normal Risk Customer.")
st.markdown("---")
st.markdown("**Model Info:** This prediction is powered by a Random Forest Regressor trained on real-world insurance data.")

st.markdown("*Disclaimer: Predictions are estimates and should be used for strategic insight only.*")