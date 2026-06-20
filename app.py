import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("crop_yield_model.pkl")

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Crop Yield Prediction System")

st.markdown("""
Predict expected crop yield using machine learning.
""")

# ==================================
# USER INPUTS
# ==================================

crop_type = st.selectbox(
    "Select Crop Type",
    ["corn", "rice", "soybean", "wheat", "cotton"]
)

soil_type = st.selectbox(
    "Select Soil Type",
    ["Loamy", "Clay", "Sandy", "Silt"]
)

season = st.selectbox(
    "Select Season",
    ["Spring", "Summer", "Autumn", "Winter"]
)

irrigation = st.selectbox(
    "Irrigation Applied?",
    ["Yes", "No"]
)

fertilized = st.selectbox(
    "Fertilizer Applied?",
    ["Yes", "No"]
)

# ==================================
# PREDICT BUTTON
# ==================================

if st.button("Predict Crop Yield"):

    input_data = pd.DataFrame({

        'rainfall_mm': [1200],
        'avg_temp_c': [28],
        'humidity_pct': [70],
        'soil_ph': [6.5],
        'nitrogen_kg_ha': [120],
        'phosphorus_kg_ha': [60],
        'potassium_kg_ha': [80],
        'irrigation_mm': [500 if irrigation == "Yes" else 0],
        'pest_index': [2],
        'days_from_last_harvest': [60],

        'weather_zone': ['Tropical'],
        'soil_type': [soil_type],
        'season': [season],
        'crop_type': [crop_type],
        'irrigation_method': [
            'Drip' if irrigation == "Yes" else 'None'
        ],
        'fertilized': [fertilized],
        'seed_quality': ['Medium']
    })

    st.subheader("Farm Information")
    prediction = model.predict(input_data)[0]
    

    st.metric(
    label="Estimated Yield",
    value=f"{prediction:.2f} tons/hectare"
    )

    if prediction < 15:
    st.error("Low Yield Expected")

    elif prediction < 25:
    st.warning("Moderate Yield Expected")

    else:
    st.success("High Yield Expected")
