import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — every text element has an explicit dark colour
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ── Base background ── */
[data-testid="stAppViewContainer"] { background: #f5f7f2; }

/* ── All body text dark by default ── */
html, body, [class*="css"], p, span, div, li, td, th, label,
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
    color: #1a1a1a;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1b4332; }
[data-testid="stSidebar"],
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] em,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown *,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown em,
[data-testid="stSidebar"] .stMarkdown strong,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
    color: #d8f3dc !important;
}
[data-testid="stSidebar"] hr { border-color: #40916c55; }
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span {
    font-size: 1rem;
    padding: 6px 0;
    color: #d8f3dc !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #d8f3dc;
    border-left: 5px solid #40916c;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #2d6a4f !important;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricValue"],
[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    color: #1b4332 !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"],
[data-testid="metric-container"] [data-testid="stMetricDelta"] p {
    color: #52796f !important;
}

/* ── Section cards ── */
.section-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    margin-bottom: 20px;
    color: #1a1a1a;
}
.section-card p, .section-card li,
.section-card span, .section-card td,
.section-card th { color: #1a1a1a !important; }
.section-card h3 { color: #1b4332 !important; }

/* ── Page header (gradient banner) ── */
.page-header {
    background: linear-gradient(135deg, #1b4332 0%, #40916c 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
    color: #ffffff !important;
}
.page-header *   { color: #ffffff !important; }
.page-header h1  { color: #ffffff !important; margin: 0; font-size: 2rem; }
.page-header p   { color: #ffffff !important; margin: 6px 0 0 0; font-size: 1rem; }

/* ── Yield band badges ── */
.badge-high { background:#d8f3dc; color:#1b4332 !important; padding:6px 14px; border-radius:20px; font-weight:700; display:inline-block; }
.badge-mid  { background:#fff3cd; color:#7d5a00 !important; padding:6px 14px; border-radius:20px; font-weight:700; display:inline-block; }
.badge-low  { background:#f8d7da; color:#842029 !important; padding:6px 14px; border-radius:20px; font-weight:700; display:inline-block; }

/* ── Recommendation band header ── */
.rec-band {
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* ── Input labels ── */
label, .stSelectbox label, .stSlider label,
.stExpander label, div[data-testid="stFormLabel"] {
    font-weight: 600 !important;
    color: #2d6a4f !important;
}

/* ── Slider value text ── */
[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
    color: #1a1a1a !important;
}

/* ── Expander header text ── */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary {
    color: #1b4332 !important;
    font-weight: 600;
}
[data-testid="stExpander"] { border: 1px solid #d8f3dc; border-radius: 10px; }

/* ── Markdown headings inside main area ── */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5 {
    color: #1b4332 !important;
}

/* ── Markdown body text ── */
.stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th {
    color: #1a1a1a !important;
}

/* ── Tables in markdown ── */
.stMarkdown table { width: 100%; border-collapse: collapse; }
.stMarkdown table th {
    background: #1b4332 !important;
    color: #ffffff !important;
    padding: 8px 12px;
}
.stMarkdown table td {
    color: #1a1a1a !important;
    padding: 7px 12px;
    border-bottom: 1px solid #e0e0e0;
}
.stMarkdown table tr:nth-child(even) td { background: #f5f7f2; }

/* ── st.caption ── */
[data-testid="stCaptionContainer"] p,
.stCaption, .stCaption p { color: #52796f !important; font-size: 0.85rem; }

/* ── st.info / st.success / st.warning / st.error ── */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div { color: #1a1a1a !important; }

/* ── Tabs ── */
[data-baseweb="tab"] { font-weight: 600; color: #1a1a1a !important; }
[aria-selected="true"][data-baseweb="tab"] { color: #2d6a4f !important; border-bottom-color: #2d6a4f !important; }
[data-baseweb="tab-panel"] p,
[data-baseweb="tab-panel"] span { color: #1a1a1a !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #2d6a4f, #52b788);
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    padding: 12px 32px;
    font-size: 1.05rem;
    font-weight: 700;
    width: 100%;
    cursor: pointer;
    transition: opacity .2s;
}
div.stButton > button:hover { opacity: 0.88; }
div.stButton > button p,
div.stButton > button span { color: #ffffff !important; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #40916c !important;
    color: #ffffff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
}
[data-testid="stDownloadButton"] > button p,
[data-testid="stDownloadButton"] > button span { color: #ffffff !important; }

/* ── Selectbox / dropdown — selected value and all options ── */
[data-baseweb="select"] span,
[data-baseweb="select"] div,
[data-baseweb="select"] input,
[data-baseweb="select"] [class*="placeholder"],
[data-baseweb="select"] [class*="singleValue"],
[data-baseweb="select"] [class*="ValueContainer"] *,
[data-baseweb="select"] [class*="control"] *,
div[data-testid="stSelectbox"] div,
div[data-testid="stSelectbox"] span,
div[data-testid="stSelectbox"] input,
/* Dropdown menu options */
[data-baseweb="menu"] li,
[data-baseweb="menu"] li *,
[data-baseweb="menu"] [role="option"],
[data-baseweb="menu"] [role="option"] *,
ul[role="listbox"] li,
ul[role="listbox"] li * {
    color: #1a1a1a !important;
    background-color: #ffffff;
}
/* Hover state on dropdown options */
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"]:hover * {
    background-color: #d8f3dc !important;
    color: #1b4332 !important;
}

/* ── Radio button text ── */
.stRadio div label span { color: #d8f3dc !important; }

/* ── st.markdown bold ── */
strong { color: inherit !important; }

</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# LOAD ASSETS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model.pkl")

@st.cache_data
def load_results():
    try:
        return pd.read_csv("actual_vs_predicted.csv")
    except Exception:
        return None

@st.cache_data
def load_feature_importance():
    try:
        return pd.read_csv("feature_importance.csv")
    except Exception:
        return None

try:
    model = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

actual_pred_df = load_results()
feature_imp_df = load_feature_importance()

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  CropYield AI")
    st.markdown("*Intelligent Yield Forecasting System*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Prediction", "Model Performance"]
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Model:** Tuned XGBoost")
    st.markdown("**R² Score:** 0.8429")
    st.markdown("**Accuracy:** 84.29%")
    st.markdown("---")
    st.markdown(
        "<small style='color:#d8f3dc'>© 2026 Smart Crop Yield Predictor<br>Final Year Project</small>",
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: GAUGE CHART
# ──────────────────────────────────────────────────────────────────────────────
def make_gauge(value):
    color = "#e63946" if value < 15 else ("#f4a261" if value < 27 else "#52b788")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": 23.72, "valueformat": ".2f"},
        title={"text": "Predicted Yield (tons/ha)", "font": {"size": 18, "color": "#1b4332"}},
        number={"suffix": " t/ha", "font": {"size": 32, "color": "#1b4332"}},
        gauge={
            "axis": {"range": [0, 41], "tickwidth": 1, "tickcolor": "#555",
                     "tickfont": {"color": "#333"}},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor": "#f5f7f2",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  15], "color": "#fde8e9"},
                {"range": [15, 27], "color": "#fff3cd"},
                {"range": [27, 41], "color": "#d8f3dc"},
            ],
            "threshold": {
                "line": {"color": "#1b4332", "width": 3},
                "thickness": 0.8,
                "value": 23.72
            }
        }
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1a1a1a"}
    )
    return fig

# ──────────────────────────────────────────────────────────────────────────────
# HELPER: RECOMMENDATIONS
# ──────────────────────────────────────────────────────────────────────────────
RECS = {
    "low": {
        "corn":    ["Apply 140–160 kg/ha of nitrogen fertiliser before next planting","Increase irrigation to at least 600 mm/season","Test soil pH and lime if below 5.8","Use certified hybrid seed varieties for improved germination"],
        "rice":    ["Ensure paddy fields maintain 5–10 cm water depth during tillering","Apply 100–120 kg/ha nitrogen in split doses","Check for blast and stem borer infestation","Transplant at 21 days to reduce establishment stress"],
        "soybean": ["Inoculate seeds with Bradyrhizobium to enhance nitrogen fixation","Apply 20–40 kg/ha phosphorus to support root development","Ensure soil pH is between 6.0 and 6.8","Control aphid populations early in the season"],
        "wheat":   ["Apply 100–120 kg/ha nitrogen at sowing and topdress at tillering","Ensure adequate soil moisture at germination stage","Use disease-resistant certified wheat varieties","Control rust and septoria leaf blotch proactively"],
        "cotton":  ["Apply 80–100 kg/ha nitrogen in split applications","Ensure irrigation at flowering and boll formation stages","Scout for bollworm and apply targeted control if threshold exceeded","Avoid waterlogging — cotton is highly sensitive to root saturation"],
        "default": ["Apply balanced NPK fertiliser before next season","Increase irrigation frequency during critical growth stages","Conduct soil test to identify deficiency areas","Use certified seeds for improved yield potential"]
    },
    "moderate": {
        "corn":    ["Topdress with 60–80 kg/ha nitrogen at the V6 growth stage","Monitor for fall armyworm — apply control if >1 larva per plant","Maintain irrigation at 500–600 mm/season","Consider potassium supplementation if soil K is below 120 kg/ha"],
        "rice":    ["Maintain water depth consistently at 5 cm during reproductive stage","Apply potassium at 60 kg/ha before panicle initiation","Monitor for bacterial leaf blight following heavy rainfall","Harvest at 80–85% grain maturity to minimise post-harvest losses"],
        "soybean": ["Apply 40 kg/ha phosphorus at flowering to support pod fill","Maintain consistent soil moisture during R3–R6 growth stages","Monitor for soybean rust and apply fungicide at first sign","Avoid herbicide stress during the critical V3–V5 window"],
        "wheat":   ["Topdress with 40–60 kg/ha nitrogen at Feekes scale stage 5","Ensure soil moisture at booting and heading stages","Apply foliar micronutrient blend if leaf chlorosis is observed","Schedule harvest within 7 days of physiological maturity"],
        "cotton":  ["Maintain irrigation at 400–500 mm through boll development","Apply 40–60 kg/ha potassium to support fibre development","Scout weekly for whitefly and thrips from squaring onwards","Apply growth regulator at first flower if vegetative growth is excessive"],
        "default": ["Maintain current NPK application rates","Monitor soil moisture levels weekly","Scout for pest pressure fortnightly","Review seed quality before the next planting season"]
    },
    "high": {
        "corn":    ["Maintain nitrogen at 140 kg/ha — avoid over-application above 180 kg/ha","Continue irrigation scheduling at current rate","Record cultivar and input data for replication next season","Consider precision monitoring tools to maintain spatial yield consistency"],
        "rice":    ["Maintain water management protocols — consistency is key at this yield level","Continue current NPK programme","Introduce post-harvest drying to <14% moisture to preserve grain quality","Document field conditions for next season's planning"],
        "soybean": ["Continue Bradyrhizobium inoculation each season","Maintain phosphorus application — do not reduce below 40 kg/ha","Monitor for late-season charcoal rot under dry conditions","Record pest and disease observations for integrated pest management planning"],
        "wheat":   ["Maintain nitrogen and irrigation programme","Introduce precision yield mapping to identify within-field variation","Review varietal performance — consider upgrading to newer certified varieties","Plan early harvest logistics to avoid lodging losses"],
        "cotton":  ["Continue current irrigation and nutrition management","Maintain bollworm monitoring even at high yield — threshold management is still essential","Document fibre quality results alongside yield for market planning","Consider cover cropping to maintain soil organic matter between seasons"],
        "default": ["Continue current best practices","Document inputs and field conditions for next season","Introduce precision agriculture monitoring to maintain this yield level","Explore certified seed upgrades for the next crop cycle"]
    }
}

def get_recommendations(prediction, crop):
    band = "low" if prediction < 15 else ("moderate" if prediction < 27 else "high")
    recs = RECS[band].get(crop.lower(), RECS[band]["default"])
    return band, recs

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Prediction":

    st.markdown("""
    <div class="page-header">
        <h1>🌾 Smart Crop Yield Prediction</h1>
        <p>Enter your farm parameters below to receive an AI-powered yield forecast and agronomic recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🌱 Farm Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        crop_type = st.selectbox("Crop Type", ["corn", "rice", "soybean", "wheat", "cotton"])
        season    = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
        soil_type = st.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Silt"])
    with col2:
        fertilized       = st.selectbox("Fertilizer Applied?", ["Yes", "No"])
        irrigation_input = st.selectbox("Irrigation Method", ["Drip", "Sprinkler", "Flood", "None"])
        seed_quality     = st.selectbox("Seed Quality", ["Certified", "Improved", "Local"])
    with col3:
        rainfall_mm  = st.slider("Rainfall (mm/season)", 200, 2000, 1200, 50)
        avg_temp_c   = st.slider("Average Temperature (°C)", 10, 45, 28, 1)
        humidity_pct = st.slider("Humidity (%)", 30, 100, 70, 5)

    with st.expander("⚙️ Advanced Soil Parameters (optional — defaults are zone averages)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            soil_ph        = st.slider("Soil pH", 4.0, 9.0, 6.5, 0.1)
            nitrogen_kg_ha = st.slider("Nitrogen (kg/ha)", 0, 300, 120, 10)
        with c2:
            phosphorus_kg_ha = st.slider("Phosphorus (kg/ha)", 0, 200, 60, 5)
            potassium_kg_ha  = st.slider("Potassium (kg/ha)", 0, 300, 80, 10)
        with c3:
            irrigation_mm  = st.slider("Irrigation Applied (mm)", 0, 1500, 500, 50)
            pest_index     = st.slider("Pest Pressure Index (0–10)", 0, 10, 2, 1)
            days_from_harvest = st.slider("Days Since Last Harvest", 10, 365, 60, 5)

    st.markdown('</div>', unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    predict_clicked = st.button("🔍 Predict Crop Yield", use_container_width=True)

    if predict_clicked:
        if not model_loaded:
            st.error("Model file not found. Please ensure crop_yield_model.pkl is present.")
        else:
            input_data = pd.DataFrame({
                'rainfall_mm':            [rainfall_mm],
                'avg_temp_c':             [avg_temp_c],
                'humidity_pct':           [humidity_pct],
                'soil_ph':                [soil_ph],
                'nitrogen_kg_ha':         [nitrogen_kg_ha],
                'phosphorus_kg_ha':       [phosphorus_kg_ha],
                'potassium_kg_ha':        [potassium_kg_ha],
                'irrigation_mm':          [irrigation_mm],
                'pest_index':             [pest_index],
                'days_from_last_harvest': [days_from_harvest],
                'weather_zone':           ['Tropical'],
                'soil_type':              [soil_type],
                'season':                 [season],
                'crop_type':              [crop_type],
                'irrigation_method':      [irrigation_input],
                'fertilized':             [fertilized],
                'seed_quality':           [seed_quality],
            })

            prediction = model.predict(input_data)[0]
            band, recs = get_recommendations(prediction, crop_type)

            # ── Result card ──
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 📊 Prediction Result")

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.plotly_chart(make_gauge(prediction), use_container_width=True)

            with res_col2:
                st.markdown("<br>", unsafe_allow_html=True)

                badge_html = {
                    "low":      '<span class="badge-low">🔴 Low Yield Expected</span>',
                    "moderate": '<span class="badge-mid">🟡 Moderate Yield Expected</span>',
                    "high":     '<span class="badge-high">🟢 High Yield Expected</span>',
                }
                st.markdown(badge_html[band], unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                m1, m2 = st.columns(2)
                m1.metric("Predicted Yield", f"{prediction:.2f} t/ha")
                m2.metric("vs Dataset Mean", f"{prediction - 23.72:+.2f} t/ha")
                m3, m4 = st.columns(2)
                m3.metric("Crop", crop_type.capitalize())
                m4.metric("Season", season)

            st.markdown('</div>', unsafe_allow_html=True)

            # ── Recommendations ──
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### 💡 Smart Agronomic Recommendations")

            band_styles = {
                "low":      ("🔴 Low Yield Band (< 15 t/ha)",      "#f8d7da", "#842029"),
                "moderate": ("🟡 Moderate Yield Band (15–27 t/ha)", "#fff3cd", "#7d5a00"),
                "high":     ("🟢 High Yield Band (> 27 t/ha)",      "#d8f3dc", "#1b4332"),
            }
            label, bg, fg = band_styles[band]
            st.markdown(
                f'<div class="rec-band" style="background:{bg};color:{fg};">{label}</div>',
                unsafe_allow_html=True
            )
            for i, rec in enumerate(recs, 1):
                st.markdown(
                    f'<p style="color:#1a1a1a;margin:6px 0"><strong style="color:#1b4332">{i}.</strong> {rec}</p>',
                    unsafe_allow_html=True
                )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Download ──
            report = f"""SMART CROP YIELD PREDICTION REPORT
====================================
Generated by: CropYield AI System
Academic Year: 2025/2026

FARM PARAMETERS
---------------
Crop Type         : {crop_type.capitalize()}
Soil Type         : {soil_type}
Season            : {season}
Irrigation Method : {irrigation_input}
Fertilizer        : {fertilized}
Seed Quality      : {seed_quality}
Rainfall (mm)     : {rainfall_mm}
Temperature (°C)  : {avg_temp_c}
Humidity (%)      : {humidity_pct}
Soil pH           : {soil_ph}
Nitrogen (kg/ha)  : {nitrogen_kg_ha}
Phosphorus (kg/ha): {phosphorus_kg_ha}
Potassium (kg/ha) : {potassium_kg_ha}
Irrigation (mm)   : {irrigation_mm}
Pest Index        : {pest_index}

PREDICTION RESULT
-----------------
Predicted Yield   : {prediction:.2f} tons/hectare
Yield Band        : {band.capitalize()}
Dataset Mean      : 23.72 tons/hectare
Difference        : {prediction - 23.72:+.2f} tons/hectare

AGRONOMIC RECOMMENDATIONS
-------------------------
{chr(10).join(f'{i}. {r}' for i, r in enumerate(recs, 1))}

====================================
Model : Tuned XGBoost Regressor
R²    : 0.8429  |  MAE: 1.3793  |  RMSE: 1.7504
"""
            st.download_button(
                "📥 Download Full Prediction Report",
                data=report,
                file_name=f"yield_report_{crop_type}_{season}.txt",
                mime="text/plain",
                use_container_width=True
            )

            st.session_state.history.append({
                "Crop": crop_type.capitalize(),
                "Season": season,
                "Soil": soil_type,
                "Irrigation": irrigation_input,
                "Fertilized": fertilized,
                "Predicted Yield (t/ha)": round(prediction, 2),
                "Band": band.capitalize()
            })

    if st.session_state.get("history"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🕓 Prediction History")
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        st.download_button(
            "📥 Download History as CSV",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.markdown("""
    <div class="page-header">
        <h1>📊 Model Performance Dashboard</h1>
        <p>Evaluation metrics, visualisations, and model comparison for the Tuned XGBoost crop yield prediction model</p>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🏆 Best Model: Tuned XGBoost — Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² Score",  "0.8429",      "Best model")
    m2.metric("MAE",       "1.3793 t/ha", "Mean Absolute Error")
    m3.metric("RMSE",      "1.7504 t/ha", "Root Mean Square Error")
    m4.metric("Accuracy",  "84.29%",      "Variance explained")
    st.markdown('</div>', unsafe_allow_html=True)

    # Model comparison
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🤖 Model Comparison")

    comparison_df = pd.DataFrame({
        "Model": ["Random Forest", "Extra Trees", "XGBoost (Default)", "Tuned XGBoost"],
        "MAE":   [1.5616, 1.5216, 1.3929, 1.3793],
        "RMSE":  [1.9709, 1.9611, 1.7680, 1.7504],
        "R²":    [0.8008, 0.8011, 0.8397, 0.8429],
    })
    colours = ["#74c69d", "#52b788", "#2d6a4f", "#1b4332"]

    tab1, tab2, tab3 = st.tabs(["R² Score", "MAE", "RMSE"])
    common = dict(plot_bgcolor="white", paper_bgcolor="white",
                  showlegend=False, height=380,
                  font=dict(color="#1a1a1a"),
                  xaxis=dict(tickfont=dict(color="#1a1a1a")),
                  yaxis=dict(tickfont=dict(color="#1a1a1a")))

    with tab1:
        fig = go.Figure(go.Bar(x=comparison_df["Model"], y=comparison_df["R²"],
            marker_color=colours, text=comparison_df["R²"].round(4), textposition="outside",
            textfont=dict(color="#1a1a1a")))
        fig.update_layout(title="R² Score by Model (higher is better)",
            yaxis=dict(range=[0.78, 0.86], title="R² Score", tickfont=dict(color="#1a1a1a")),
            **{k:v for k,v in common.items() if k != "yaxis"})
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure(go.Bar(x=comparison_df["Model"], y=comparison_df["MAE"],
            marker_color=colours[::-1], text=comparison_df["MAE"].round(4), textposition="outside",
            textfont=dict(color="#1a1a1a")))
        fig.update_layout(title="Mean Absolute Error by Model (lower is better)",
            yaxis=dict(range=[1.3, 1.65], title="MAE (t/ha)", tickfont=dict(color="#1a1a1a")),
            **{k:v for k,v in common.items() if k != "yaxis"})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure(go.Bar(x=comparison_df["Model"], y=comparison_df["RMSE"],
            marker_color=colours[::-1], text=comparison_df["RMSE"].round(4), textposition="outside",
            textfont=dict(color="#1a1a1a")))
        fig.update_layout(title="Root Mean Square Error by Model (lower is better)",
            yaxis=dict(range=[1.7, 2.05], title="RMSE (t/ha)", tickfont=dict(color="#1a1a1a")),
            **{k:v for k,v in common.items() if k != "yaxis"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Actual vs Predicted
    if actual_pred_df is not None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Actual vs Predicted Yield")
        sample_df = actual_pred_df.sample(min(2000, len(actual_pred_df)), random_state=42)
        mn = float(sample_df[["Actual","Predicted"]].min().min())
        mx = float(sample_df[["Actual","Predicted"]].max().max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sample_df["Actual"], y=sample_df["Predicted"],
            mode="markers", marker=dict(color="#52b788", size=4, opacity=0.5), name="Predictions"))
        fig.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx], mode="lines",
            line=dict(color="#e63946", dash="dash", width=2), name="Perfect Prediction"))
        fig.update_layout(xaxis_title="Actual Yield (t/ha)", yaxis_title="Predicted Yield (t/ha)",
            plot_bgcolor="white", paper_bgcolor="white", height=450,
            legend=dict(x=0.02, y=0.97, font=dict(color="#1a1a1a")),
            font=dict(color="#1a1a1a"),
            xaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each point is one test-set record. Points along the dashed line = perfect prediction.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Residuals
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 📉 Residual Error Distribution")
        residuals = actual_pred_df["Actual"] - actual_pred_df["Predicted"]
        fig = go.Figure(go.Histogram(x=residuals, nbinsx=80,
            marker_color="#52b788", opacity=0.8))
        fig.add_vline(x=0, line_dash="dash", line_color="#e63946", line_width=2,
            annotation_text="Zero Error", annotation_position="top right",
            annotation_font_color="#1a1a1a")
        fig.update_layout(xaxis_title="Residual (Actual − Predicted)", yaxis_title="Frequency",
            plot_bgcolor="white", paper_bgcolor="white", height=380, showlegend=False,
            font=dict(color="#1a1a1a"),
            xaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Near-normal distribution centred at zero confirms no systematic prediction bias.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Feature importance
    if feature_imp_df is not None:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🔑 Feature Importance (XGBoost Gain Score)")
        fi = feature_imp_df.sort_values("Importance", ascending=True).tail(15)
        fig = go.Figure(go.Bar(x=fi["Importance"], y=fi["Feature"], orientation="h",
            marker=dict(color=fi["Importance"],
                colorscale=[[0,"#d8f3dc"],[0.5,"#52b788"],[1,"#1b4332"]], showscale=False)))
        fig.update_layout(xaxis_title="Importance Score (Gain)", yaxis_title="",
            plot_bgcolor="white", paper_bgcolor="white", height=420, margin=dict(l=160),
            font=dict(color="#1a1a1a"),
            xaxis=dict(tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
            yaxis=dict(tickfont=dict(color="#1a1a1a")))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Gain-based importance: how much each feature reduces prediction error across all trees.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Crop-specific
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🌱 Crop-Specific Model Performance")
    crop_df = pd.DataFrame({
        "Crop":  ["Cotton","Rice","Wheat","Soybean","Corn"],
        "MAE":   [1.2632, 1.2351, 1.4348, 1.4498, 1.5319],
        "RMSE":  [1.6029, 1.5627, 1.8198, 1.8363, 1.9397],
        "R²":    [0.8430, 0.8181, 0.8003, 0.7725, 0.7496],
    })
    bar_colours = ["#1b4332" if r >= 0.75 else "#e63946" for r in crop_df["R²"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=crop_df["Crop"], y=crop_df["R²"], marker_color=bar_colours,
        text=crop_df["R²"].round(4), textposition="outside",
        textfont=dict(color="#1a1a1a"), name="R² Score"))
    fig.add_hline(y=0.75, line_dash="dash", line_color="#f4a261", line_width=2,
        annotation_text="Satisfactory threshold (R²=0.75)", annotation_position="top left",
        annotation_font_color="#1a1a1a")
    fig.update_layout(
        yaxis=dict(range=[0.72, 0.88], title="R² Score",
                   tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
        xaxis=dict(title="Crop Type", tickfont=dict(color="#1a1a1a"), title_font=dict(color="#1a1a1a")),
        plot_bgcolor="white", paper_bgcolor="white", height=380,
        showlegend=False, font=dict(color="#1a1a1a"))
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        crop_df.style
            .highlight_max(subset=["R²"], color="#d8f3dc")
            .highlight_min(subset=["R²"], color="#fde8e9")
            .format({"MAE":"{:.4f}","RMSE":"{:.4f}","R²":"{:.4f}"}),
        use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Model summary
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ✅ Model Summary")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("""
| Property | Value |
|----------|-------|
| Algorithm | Tuned XGBoost Regressor |
| n_estimators | 500 |
| max_depth | 5 |
| learning_rate | 0.1 |
| subsample | 0.7 |
| gamma | 0.3 |
| colsample_bytree | 1.0 |
""")
    with s2:
        st.markdown("""
| Metric | Value |
|--------|-------|
| R² Score | 0.8429 |
| MAE | 1.3793 t/ha |
| RMSE | 1.7504 t/ha |
| Accuracy | 84.29% |
| Training records | 197,855 |
| Test records | 49,464 |
| Status | ✅ Production Ready |
""")
    st.markdown('</div>', unsafe_allow_html=True)
