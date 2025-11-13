import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import requests
from datetime import datetime
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import time

# ---- Force Streamlit Cloud theme overrides so CSS behaves like local ----
st.markdown("""
    <style>
        /* Make base backgrounds transparent so our gradient shows */
        .stApp, .main, .block-container, [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }
        body, html {
            background: transparent !important;
        }
        /* Slight sidebar padding fix on Cloud */
        [data-testid="stSidebar"] { padding-top: 20px !important; }
        /* Force readable text inside native select/dropdown lists */
        .stSelectbox div[data-baseweb="select"] * { color: black !important; }
    </style>
""", unsafe_allow_html=True)

# ----------------- Page Config & Styling -----------------

st.set_page_config(page_title="Weather + LSTM Dashboard", page_icon="🌤️", layout="wide")

# THIS IS THE "Dark Glass" UI theme - FINAL-FINAL-FIX
st.markdown("""
    <style>
    /* 1. Main Page Background - Dark Gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        background-size: cover;
    }
    
    /* 2. Sidebar Styling - Dark Glass */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    /* 3. Glass Effect for Metric Cards - Dark Glass */
    [data-testid="stMetric"] {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* 4. A generic "glass-container" class for plots/tables */
    .glass-container {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* 5. Style for Streamlit's Tabs */
    [data-testid="stTabs"] {
        border-bottom: 1px solid rgba(255, 255, 255, 0.15);
    }
    [data-testid="stTab"] {
        color: rgba(255, 255, 255, 0.7);
    }
    [data-testid="stTab"][aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #ffffff;
        color: #ffffff;
    }
    
    /* 6. Make all page text white for contrast */
    h1, h2, h3, h4, h5, h6, p, label, .st-bf, .st-bs, .st-bt, .st-bu {
        color: #f1f1f1;
    }
    
    /* 7. FIX for dropdown list items (make them black) - STRONGER RULE */
    [data-testid="stSelectbox-list"] * {
        color: #333333 !important; /* Force dark text for ALL elements inside dropdown */
    }
    
    /* 8. FIX: Make text inside metric cards white */
    [data-testid="stMetric"] label, [data-testid="stMetric"] div {
        color: #f1f1f1 !important; /* Force white text for label and value */
    }
    
    </style>
""", unsafe_allow_html=True)


# ----------------- Data & Model Loading -----------------

@st.cache_data
def get_path(paths):
    """Find the first valid path from a list."""
    return next((p for p in paths if os.path.exists(p)), None)

@st.cache_resource
def load_assets():
    """Load model and scaler once."""
    MODEL_PATH = get_path(["LSTM.keras", "models/LSTM.keras"])
    SCALER_PATH = get_path(["scaler.pkl", "models/scaler.pkl"])
    
    if not MODEL_PATH or not SCALER_PATH:
        st.error("Error: Model or Scaler file not found.")
        st.stop()
        
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_csv(csv_path):
    """Load the main data CSV."""
    if not csv_path:
        st.error("Error: weather_data.csv not found.")
        st.stop()
    df = pd.read_csv(csv_path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    return df

CSV_PATH = get_path(["weather_data.csv"])
model, scaler = load_assets()
df = load_csv(CSV_PATH)


# ----------------- Sidebar -----------------
st.sidebar.header("⚙️ Settings")

unique_cities = sorted(df["city"].unique().tolist())
selected_city = st.sidebar.selectbox("🌆 Select City", unique_cities)
city = selected_city

# Use an expander to hide the API key and button
with st.sidebar.expander("🔗 Fetch Live Data"):
    api_key = st.text_input("🔑 API Key", type="password", help="Get key from OpenWeatherMap")
    if st.button("📥 Fetch & Add Live Data"):
        if not api_key:
            st.warning("Enter API key first!")
        else:
            try:
                live = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}").json()
                live_row = {
                    "date_time": datetime.now(),
                    "city": live["name"],
                    "lat": live["coord"]["lat"],
                    "lon": live["coord"]["lon"],
                    "temp_celsius": round(live["main"]["temp"] - 273.15, 2),
                    "humidity": live["main"]["humidity"],
                    "wind_mps": live["wind"]["speed"],
                    "risk_level": "High" if round(live["main"]["temp"] - 273.15, 2) > 40 else "Low"
                }
                pd.DataFrame([live_row]).to_csv(CSV_PATH, mode='a', header=False, index=False)
                st.success("✅ Live data added!")
                st.cache_data.clear() # Clear cache to reload data
            except Exception as e:
                st.error(f"❌ Error: {e}")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("⏱ Auto-refresh", value=False)
if auto_refresh:
    refresh_secs = st.sidebar.slider("Refresh interval (sec)", 5, 60, 10)
    time.sleep(refresh_secs)
    st.rerun()

# Filter data for selected city
df_city = df[df["city"] == city]

# ----------------- Main View -----------------
st.title(f"🌍 Weather & LSTM Forecast: **{city}**")

if df_city.empty:
    st.warning(f"⚠️ No data found for {city}.")
    other_cities = [c for c in unique_cities if c != city]
    if other_cities:
        st.info("✅ You can view data for these cities:")
        st.write(", ".join([f"**{c}**" for c in other_cities]))
    st.stop()

# Use Tabs for a clean layout
tab1, tab2, tab3 = st.tabs(["📈 Overview & Forecast", "📄 Raw Data", "🗺️ City Map"])

# --- Tab 1: Overview & Forecast (WITH LATEST CHART UPDATES) ---
with tab1:
    st.subheader("Current Conditions")
    latest = df_city.iloc[-1]
    
    # Metrics now get the glass effect from CSS
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡 Temp (°C)", f"{latest['temp_celsius']} °C")
    col2.metric("💧 Humidity", f"{latest['humidity']} %")
    col3.metric("💨 Wind", f"{latest['wind_mps']} m/s")
    col4.metric("⚠️ Risk", latest["risk_level"])

    st.subheader("📈 Temperature Trend & Prediction")
    
    # Wrap plot in our .glass-container
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    # Prediction logic
    LOOKBACK = model.input_shape[1]
    block = df_city[['temp_celsius', 'humidity', 'wind_mps']].tail(LOOKBACK).values
    
    if len(block) < LOOKBACK:
        st.warning(f"Not enough data to predict for {city} (need {LOOKBACK} records, have {len(block)}).")
    else:
        X_input = scaler.transform(block).reshape(1, LOOKBACK, 3)
        y_pred_scaled = model.predict(X_input)
        pred_temp = scaler.inverse_transform([[y_pred_scaled[0][0], 0, 0]])[0][0]

        # --- Plot ---
        past = df_city["temp_celsius"].tail(LOOKBACK).reset_index(drop=True)
        fig = go.Figure()
        
        # Past data trace
        fig.add_trace(go.Scatter(
            y=past, 
            mode="lines+markers", 
            name="Past Temperature",
            line=dict(color='#FFFF00', width=3), # Bright yellow
            marker=dict(color='#FFFFFF', size=4)
        ))
        
        # Predicted data trace
        fig.add_trace(go.Scatter(
            x=[len(past)], 
            y=[pred_temp], 
            mode="markers+text", 
            name="Predicted Temp",
            marker=dict(color='red', size=14, symbol='star', line=dict(color='white', width=1)),
            text=[f"<b>{round(pred_temp,2)}°C</b>"],
            textposition="top center",
            textfont=dict(color='#FFFFFF', size=15)
        ))
        
        # --- NEW LAYOUT: FIXES AXIS TEXT ---
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            font_color='#f1f1f1',           # White text for titles
            xaxis_title="Time Step (Past to Future)",
            yaxis_title="Temperature (°C)",
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1,
                font=dict(color='#f1f1f1') # Ensure legend text is white
            ),
            xaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)', # Faint gridlines
                tickfont=dict(color='#f1f1f1')        # FIX: Make X-axis numbers white
            ),
            yaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)', # Faint gridlines
                tickfont=dict(color='#f1f1f1')        # FIX: Make Y-axis numbers white
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True) # Close the glass container

# --- Tab 2: Raw Data ---
with tab2:
    st.subheader(f"📄 Raw Data History for {city}")
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.dataframe(df_city.tail(20), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 3: City Map ---
with tab3:
    st.subheader(f"🗺️ Map Location")
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    
    # Get latest lat/lon
    map_data = df_city[['lat', 'lon']].iloc[-1:].copy()
    map_data['lat'] = pd.to_numeric(map_data['lat'])
    map_data['lon'] = pd.to_numeric(map_data['lon'])
    
    if not map_data.empty:
        st.map(map_data, zoom=10)
    else:
        st.info("No location data available to display on map.")
        
    st.markdown('</div>', unsafe_allow_html=True)

