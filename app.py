
import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="AQI Prediction Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = joblib.load("random_forest_aqi_model.pkl")


# -------------------------------------------------
# AQI Category Function
# -------------------------------------------------
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("🌍 Air Quality Analytics & AQI Prediction")
st.write(
    "Enter pollutant measurements to predict the Air Quality Index (AQI) "
    "using a trained Random Forest machine learning model."
)

st.divider()


# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.subheader("🧪 Pollutant Measurements")

col1, col2, col3 = st.columns(3)

with col1:
    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=120.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=180.0
    )

    no = st.number_input(
        "NO",
        min_value=0.0,
        value=40.0
    )

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=50.0
    )


with col2:
    nox = st.number_input(
        "NOx",
        min_value=0.0,
        value=70.0
    )

    nh3 = st.number_input(
        "NH3",
        min_value=0.0,
        value=30.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=1.5
    )

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=20.0
    )


with col3:
    o3 = st.number_input(
        "O3",
        min_value=0.0,
        value=60.0
    )

    benzene = st.number_input(
        "Benzene",
        min_value=0.0,
        value=3.0
    )

    toluene = st.number_input(
        "Toluene",
        min_value=0.0,
        value=8.0
    )

    xylene = st.number_input(
        "Xylene",
        min_value=0.0,
        value=2.0
    )


st.divider()


# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.button("🔍 Predict AQI", use_container_width=True):

    input_data = pd.DataFrame([{
        "PM2.5": pm25,
        "PM10": pm10,
        "NO": no,
        "NO2": no2,
        "NOx": nox,
        "NH3": nh3,
        "CO": co,
        "SO2": so2,
        "O3": o3,
        "Benzene": benzene,
        "Toluene": toluene,
        "Xylene": xylene
    }])
    # -------------------------------------------------
    # Input Validation
    # -------------------------------------------------
    if (input_data.iloc[0] == 0).all():
        st.error("Please enter valid pollutant values before predicting AQI.")
        st.stop()

    # Make prediction
    predicted_aqi = model.predict(input_data)[0]

    # Get category
    category = get_aqi_category(predicted_aqi)
    # -------------------------------------------------
    # Health Recommendation
    # -------------------------------------------------
    recommendations = {
        "Good": "Air quality is good. Outdoor activities are generally safe.",
        "Satisfactory": "Air quality is satisfactory. Sensitive individuals should take normal precautions.",
        "Moderate": "Air quality is moderate. Sensitive individuals should limit prolonged outdoor exposure.",
        "Poor": "Air quality is poor. Consider limiting prolonged outdoor activities.",
        "Very Poor": "Air quality is very poor. Avoid prolonged outdoor activities and take necessary precautions.",
        "Severe": "Air quality is severe. Avoid outdoor exposure as much as possible."
    }

    recommendation = recommendations[category]

    st.subheader("❤️ Health Recommendation")
    # -------------------------------------------------
    # AQI Visual Indicator
    # -------------------------------------------------
    st.subheader("📈 AQI Level Indicator")

    aqi_progress = min(int((predicted_aqi / 500) * 100), 100)

    st.progress(aqi_progress)

    st.caption(
        f"AQI Level: {predicted_aqi:.2f} / 500"
    )
    st.info(recommendation)


    # -------------------------------------------------
    # Prediction Result
    # -------------------------------------------------
    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            label="Predicted AQI",
            value=f"{predicted_aqi:.2f}"
        )

    with result_col2:
        st.metric(
            label="AQI Category",
            value=category
        )


    # -------------------------------------------------
    # Category Message
    # -------------------------------------------------
    if category == "Good":
        st.success("🟢 Air quality is Good.")

    elif category == "Satisfactory":
        st.success("🟢 Air quality is Satisfactory.")

    elif category == "Moderate":
        st.warning("🟡 Air quality is Moderate.")

    elif category == "Poor":
        st.warning("🟠 Air quality is Poor.")

    elif category == "Very Poor":
        st.error("🔴 Air quality is Very Poor.")

    else:
        st.error("🔴 Air quality is Severe.")


    # -------------------------------------------------
    # Input Summary
    # -------------------------------------------------
    st.subheader("📋 Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()

st.caption(
    "AQI Prediction Dashboard | Random Forest Regression | "
    "Air Quality Analytics Project"
)