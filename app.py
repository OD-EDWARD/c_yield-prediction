import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Smart Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("crop_yield_model.pkl")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌾 Crop Yield Predictor")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Performance",
        "About System"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    Machine Learning Model

    • Algorithm: Tuned XGBoost
    • R² Score: 0.8429
    • MAE: 1.3793
    • RMSE: 1.7504
    """
)

# ==========================================================
# PREDICTION PAGE
# ==========================================================

if page == "Prediction":

    st.title("🌾 Smart Crop Yield Prediction Dashboard")

    st.caption(
        "AI-powered crop yield forecasting system"
    )

    st.subheader("Farm Information")

    col1, col2 = st.columns(2)

    with col1:
        crop_type = st.selectbox(
            "Crop Type",
            ["corn", "rice", "soybean", "wheat", "cotton"]
        )

        season = st.selectbox(
            "Season",
            ["Spring", "Summer", "Autumn", "Winter"]
        )

        fertilized = st.selectbox(
            "Fertilizer Applied?",
            ["Yes", "No"]
        )

    with col2:
        soil_type = st.selectbox(
            "Soil Type",
            ["Loamy", "Clay", "Sandy", "Silt"]
        )

        irrigation = st.selectbox(
            "Irrigation Applied?",
            ["Yes", "No"]
        )

    # Prediction History
    if "history" not in st.session_state:
        st.session_state.history = []

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

        prediction = model.predict(input_data)[0]

        # Metric Card
        st.subheader("Prediction Result")

        st.metric(
            "Estimated Yield",
            f"{prediction:.2f} tons/hectare"
        )

        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Yield (tons/hectare)"},
            gauge={
                'axis': {'range': [0, 40]}
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Yield Interpretation
        if prediction < 15:

            st.error("🔴 Low Yield Expected")

            st.info(
                """
                Recommendation:
                • Improve irrigation
                • Apply fertilizer
                • Check soil quality
                """
            )

        elif prediction < 25:

            st.warning("🟡 Moderate Yield Expected")

            st.info(
                """
                Recommendation:
                • Maintain nutrient levels
                • Monitor pests regularly
                """
            )

        else:

            st.success("🟢 High Yield Expected")

            st.info(
                """
                Recommendation:
                • Continue current practices
                • Monitor crop health
                """
            )

        # Save History
        st.session_state.history.append({
            "Crop": crop_type,
            "Soil": soil_type,
            "Season": season,
            "Yield": round(prediction, 2)
        })

        # Download Report
        report = f"""
Crop Yield Prediction Report

Crop Type: {crop_type}
Soil Type: {soil_type}
Season: {season}
Irrigation: {irrigation}
Fertilized: {fertilized}

Predicted Yield:
{prediction:.2f} tons/hectare
"""

        st.download_button(
            label="📥 Download Prediction Report",
            data=report,
            file_name="crop_yield_report.txt",
            mime="text/plain"
        )

    # Prediction History
    if len(st.session_state.history) > 0:

        st.subheader("Prediction History")

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

# ==========================================================
# MODEL PERFORMANCE PAGE
# ==========================================================

elif page == "Model Performance":

    st.title("📊 Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "R² Score",
        "0.8429"
    )

    col2.metric(
        "MAE",
        "1.3793"
    )

    col3.metric(
        "RMSE",
        "1.7504"
    )

    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": [
            "Random Forest",
            "Extra Trees",
            "XGBoost",
            "Tuned XGBoost"
        ],
        "R²": [
            0.8008,
            0.8011,
            0.8397,
            0.8429
        ]
    })

    fig = px.bar(
        comparison_df,
        x="Model",
        y="R²",
        title="Model Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "About System":

    st.title("ℹ️ About This System")

    st.markdown(
        """
### Smart Crop Yield Prediction System

This application predicts crop yield using
Machine Learning techniques.

### Features

- Crop Yield Prediction
- Yield Classification
- Smart Recommendations
- Prediction History
- Downloadable Reports
- Interactive Dashboard

### Machine Learning Model

The final model used is a
Hyperparameter-Tuned XGBoost Regressor.

### Performance

- R² Score: 0.8429
- MAE: 1.3793
- RMSE: 1.7504

### Developer

Adeyinka Ademola
Computer Science
Bowen University
"""
    )
