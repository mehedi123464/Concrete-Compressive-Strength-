import streamlit as st
import joblib
import numpy as np

st.title("Ternary Blended Mortar Strength Predictor")

model_data = joblib.load("my_KNN_model.pkl")

best_knn = model_data["model"]
x_scaler = model_data["x_scaler"]

sand = st.number_input("Sand", value=975)
cement = st.number_input("Cement", value=800)
fly_ash = st.number_input("Fly Ash", value=100)
silica_fume = st.number_input("Silica Fume", value=80)
sp = st.number_input("Superplasticizer", value=2.0)
retarder = st.number_input("Retarder", value=1.0)
accelerator = st.number_input("Accelerator", value=1.0)
wc = st.number_input("W/C Ratio", value=0.33)

if st.button("Predict"):

    features = np.array([[
        sand,
        cement,
        fly_ash,
        silica_fume,
        sp,
        retarder,
        accelerator,
        wc
    ]])

    X_scaled = x_scaler.transform(features)

    prediction = best_knn.predict(X_scaled)[0]

    st.success(f"Predicted Strength: {prediction:.2f} MPa")
