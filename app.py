import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AQI Analytics Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 18px;
    color: #666666;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 600;
    margin-top: 15px;
}

.footer {
    text-align: center;
    color: #777777;
    font-size: 14px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("random_forest_aqi_model.pkl")


model = load_model()

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("city_day.csv")

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    return df


air_quality = load_data()

# --------------------------------------------------
# AQI CATEGORY FUNCTION
# --------------------------------------------------

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


# --------------------------------------------------
# HEALTH RECOMMENDATIONS
# --------------------------------------------------

recommendations = {

    "Good":
        "Air quality is good. Outdoor activities are generally safe.",

    "Satisfactory":
        "Air quality is satisfactory. Sensitive individuals should take normal precautions.",

    "Moderate":
        "Air quality is moderate. Sensitive individuals should limit prolonged outdoor exposure.",

    "Poor":
        "Air quality is poor. Consider limiting prolonged outdoor activities.",

    "Very Poor":
        "Air quality is very poor. Avoid prolonged outdoor activities and take necessary precautions.",

    "Severe":
        "Air quality is severe. Avoid outdoor exposure as much as possible."
}

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("🌍 About This Project")

    st.write(
        "This application analyzes air quality data "
        "and predicts AQI using a Random Forest "
        "Regression model."
    )

    st.divider()

    st.subheader("🤖 Machine Learning")

    st.write("Model: Random Forest Regression")
    st.write("R² Score: 0.907")
    st.write("RMSE: 41.22")

    st.divider()

    st.subheader("📊 Dataset")

    st.write(
        f"Records: {len(air_quality):,}"
    )

    st.write(
        f"Columns: {air_quality.shape[1]}"
    )

    st.divider()

    st.subheader("🧪 Pollutants")

    st.write(
        "PM2.5, PM10, NO, NO2, NOx, NH3, CO, "
        "SO2, O3, Benzene, Toluene and Xylene"
    )

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🌍 Air Quality Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Air Quality Analysis & AQI Prediction Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '📌 Dataset Overview'
    '</div>',
    unsafe_allow_html=True
)

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:

    st.metric(
        "Total Records",
        f"{len(air_quality):,}"
    )

with metric2:

    st.metric(
        "Cities",
        air_quality["City"].nunique()
    )

with metric3:

    st.metric(
        "Average AQI",
        f"{air_quality['AQI'].mean():.2f}"
    )

with metric4:

    st.metric(
        "Maximum AQI",
        f"{air_quality['AQI'].max():.2f}"
    )

st.divider()

# --------------------------------------------------
# CITY SELECTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🏙️ City-wise Air Quality Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Select a city to view its air quality statistics and AQI trend."
)

cities = sorted(
    air_quality["City"]
    .dropna()
    .unique()
)

selected_city = st.selectbox(
    "Select City",
    cities
)

# Filter selected city
city_data = air_quality[
    air_quality["City"] == selected_city
].copy()

# --------------------------------------------------
# CITY METRICS
# --------------------------------------------------

city_metric1, city_metric2, city_metric3, city_metric4 = st.columns(4)

with city_metric1:

    st.metric(
        "Selected City",
        selected_city
    )

with city_metric2:

    st.metric(
        "Average AQI",
        f"{city_data['AQI'].mean():.2f}"
    )

with city_metric3:

    st.metric(
        "Maximum AQI",
        f"{city_data['AQI'].max():.2f}"
    )

with city_metric4:

    st.metric(
        "Records",
        f"{len(city_data):,}"
    )

# --------------------------------------------------
# CITY AQI CATEGORY
# --------------------------------------------------

city_average_aqi = city_data["AQI"].mean()

city_category = get_aqi_category(
    city_average_aqi
)

st.write("")

st.info(
    f"📍 **{selected_city}** has an average AQI of "
    f"**{city_average_aqi:.2f}**, which falls under the "
    f"**{city_category}** category."
)

# --------------------------------------------------
# CITY AQI TREND
# --------------------------------------------------

st.subheader(
    f"📈 AQI Trend - {selected_city}"
)

city_trend = (
    city_data[
        ["Date", "AQI"]
    ]
    .dropna()
    .sort_values("Date")
)

st.line_chart(
    city_trend,
    x="Date",
    y="AQI"
)

st.divider()

# --------------------------------------------------
# AQI PREDICTION
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    '🔮 AQI Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter pollutant concentrations to predict the Air Quality Index."
)

col1, col2, col3 = st.columns(3)

# --------------------------------------------------
# COLUMN 1
# --------------------------------------------------

with col1:

    st.markdown("### 🌫️ Particulate & Nitrogen")

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

# --------------------------------------------------
# COLUMN 2
# --------------------------------------------------

with col2:

    st.markdown("### 🧪 Other Pollutants")

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

# --------------------------------------------------
# COLUMN 3
# --------------------------------------------------

with col3:

    st.markdown("### 🧬 Organic & Ozone")

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

st.write("")

# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

predict_button = st.button(
    "🔍 Predict AQI",
    use_container_width=True,
    type="primary"
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

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

    if (input_data.iloc[0] == 0).all():

        st.error(
            "Please enter valid pollutant values before predicting AQI."
        )

        st.stop()

    predicted_aqi = model.predict(
        input_data
    )[0]

    category = get_aqi_category(
        predicted_aqi
    )

    recommendation = recommendations[
        category
    ]

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.subheader("📊 Prediction Result")

    result1, result2, result3 = st.columns(3)

    with result1:

        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.2f}"
        )

    with result2:

        st.metric(
            "AQI Category",
            category
        )

    with result3:

        st.metric(
            "Model",
            "Random Forest"
        )

    # --------------------------------------------------
    # AQI STATUS
    # --------------------------------------------------

    if category == "Good":

        st.success(
            "🟢 Air quality is Good."
        )

    elif category == "Satisfactory":

        st.success(
            "🟢 Air quality is Satisfactory."
        )

    elif category == "Moderate":

        st.warning(
            "🟡 Air quality is Moderate."
        )

    elif category == "Poor":

        st.warning(
            "🟠 Air quality is Poor."
        )

    elif category == "Very Poor":

        st.error(
            "🔴 Air quality is Very Poor."
        )

    else:

        st.error(
            "🔴 Air quality is Severe."
        )

    # --------------------------------------------------
    # AQI INDICATOR
    # --------------------------------------------------

    st.subheader("📈 AQI Level Indicator")

    aqi_progress = min(
        int((predicted_aqi / 500) * 100),
        100
    )

    st.progress(
        aqi_progress
    )

    st.caption(
        f"AQI Level: {predicted_aqi:.2f} / 500"
    )

    # --------------------------------------------------
    # HEALTH RECOMMENDATION
    # --------------------------------------------------

    st.subheader("❤️ Health Recommendation")

    st.info(
        recommendation
    )

    # --------------------------------------------------
    # INPUT SUMMARY
    # --------------------------------------------------

    st.subheader("📋 Input Summary")

    st.dataframe(
        input_data,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    '📊 Air Quality Analytics'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Explore patterns and trends from the air quality dataset."
)

# --------------------------------------------------
# AQI DISTRIBUTION
# --------------------------------------------------

st.subheader("📊 AQI Distribution")

aqi_data = air_quality[
    "AQI"
].dropna()

aqi_bins = pd.cut(
    aqi_data,
    bins=10
)

aqi_distribution = (
    aqi_bins
    .value_counts()
    .sort_index()
    .reset_index()
)

aqi_distribution.columns = [
    "AQI Range",
    "Count"
]

aqi_distribution["AQI Range"] = (
    aqi_distribution["AQI Range"]
    .astype(str)
)

st.bar_chart(
    aqi_distribution,
    x="AQI Range",
    y="Count"
)

# --------------------------------------------------
# TOP 10 CITIES
# --------------------------------------------------

st.subheader(
    "🏙️ Top 10 Cities by Average AQI"
)

city_avg_aqi_chart = (
    air_quality
    .groupby("City")["AQI"]
    .mean()
    .dropna()
    .sort_values(
        ascending=False
    )
    .head(10)
    .reset_index()
)

city_avg_aqi_chart.columns = [
    "City",
    "Average AQI"
]

st.bar_chart(
    city_avg_aqi_chart,
    x="City",
    y="Average AQI"
)

# --------------------------------------------------
# YEAR-WISE AQI
# --------------------------------------------------

st.subheader(
    "📈 Year-wise Average AQI"
)

year_data = air_quality[
    ["Date", "AQI"]
].dropna().copy()

year_data["Year"] = (
    year_data["Date"]
    .dt.year
)

yearly_aqi = (
    year_data
    .groupby("Year")["AQI"]
    .mean()
    .reset_index()
)

yearly_aqi.columns = [
    "Year",
    "Average AQI"
]

st.line_chart(
    yearly_aqi,
    x="Year",
    y="Average AQI"
)

# --------------------------------------------------
# MONTHLY AQI
# --------------------------------------------------

st.subheader(
    "📅 Monthly Average AQI"
)

month_data = air_quality[
    ["Date", "AQI"]
].dropna().copy()

month_data["Month"] = (
    month_data["Date"]
    .dt.month
)

monthly_aqi = (
    month_data
    .groupby("Month")["AQI"]
    .mean()
    .reset_index()
)

monthly_aqi.columns = [
    "Month",
    "Average AQI"
]

st.line_chart(
    monthly_aqi,
    x="Month",
    y="Average AQI"
)

# --------------------------------------------------
# AQI CATEGORY DISTRIBUTION
# --------------------------------------------------

st.subheader(
    "🟢 AQI Category Distribution"
)

category_counts = (
    air_quality["AQI_Bucket"]
    .value_counts()
    .reset_index()
)

category_counts.columns = [
    "AQI Category",
    "Count"
]

st.bar_chart(
    category_counts,
    x="AQI Category",
    y="Count"
)

# --------------------------------------------------
# POLLUTANT AVERAGE
# --------------------------------------------------

st.subheader(
    "🧪 Average Pollutant Concentration"
)

pollutants = [
    "PM2.5",
    "PM10",
    "NO",
    "NO2",
    "NOx",
    "NH3",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "Xylene"
]

pollutant_average = (
    air_quality[pollutants]
    .mean()
    .sort_values(
        ascending=False
    )
    .reset_index()
)

pollutant_average.columns = [
    "Pollutant",
    "Average Concentration"
]

st.bar_chart(
    pollutant_average,
    x="Pollutant",
    y="Average Concentration"
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="footer">'
    '🌍 Air Quality Analytics & AQI Prediction '
    '| Random Forest Regression '
    '| Machine Learning Project'
    '</div>',
    unsafe_allow_html=True
)