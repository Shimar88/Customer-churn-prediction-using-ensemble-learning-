import streamlit as st
import joblib
import numpy as np

model = joblib.load("../models/churn_model.pkl")

st.title("📊 Customer Churn Prediction")

tenure = st.slider("Tenure", 0, 72)
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

if st.button("Predict"):
    input_data = np.array([[tenure, monthly, total]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer will stay ✅")
