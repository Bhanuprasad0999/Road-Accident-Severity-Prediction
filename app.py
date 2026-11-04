
import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="ROAD ACCIDENT SEVERITY PREDICTION",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS Styling
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #eff6ff, #e0f2fe, #f0fdf4);
    color: #1e293b;
}

/* Main Title */
h1 {
    color: #1e3a8a !important;
    text-align: center;
    font-weight: 800 !important;
}

/* Section Headings */
h2, h3, h4 {
    color: #0369a1 !important;
    font-weight: 700 !important;
}

/* All Input Labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Dropdown Text */
[data-baseweb="select"] * {
    color: #111827 !important;
}

/* Number Input Text */
[data-baseweb="input"] input {
    color: #111827 !important;
}

/* Input Background */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background-color: #ffffff !important;
}

/* Normal Text */
.stMarkdown p {
    color: #334155;
}

/* Prediction Button */
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0891b2);
    color: #ffffff !important;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    padding: 14px;
    border: none;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0e7490);
    color: #ffffff !important;
    border: none;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e0f2fe;
}

/* Divider */
hr {
    border-color: #93c5fd !important;
}

</style>
""", unsafe_allow_html=True)


# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Main Title
st.title("🚗 ROAD ACCIDENT SEVERITY PREDICTION")

st.markdown(
    """
    <h4 style="
        text-align: center;
        color: #475569 !important;
    ">
        AI-Powered Road Safety Analysis & Prediction
    </h4>
    """,
    unsafe_allow_html=True
)

st.divider()


# Input Columns
col1, col2 = st.columns(2, gap="large")


# Left Column
with col1:

    st.subheader("🛣️ ROAD & ACCIDENT DETAILS")

    Weather_Condition = st.selectbox(
        "Weather Condition",
        ["Clear", "Rain", "Fog", "Snow", "Storm"]
    )

    Road_Surface_Condition = st.selectbox(
        "Road Surface Condition",
        ["Dry", "Wet", "Snow", "Icy"]
    )

    Light_Condition = st.selectbox(
        "Light Condition",
        ["Daylight", "Darkness", "Dawn", "Dusk"]
    )

    Road_Type = st.selectbox(
        "Road Type",
        [
            "Highway",
            "Urban Road",
            "Rural Road",
            "Residential Road"
        ]
    )

    Junction_Type = st.selectbox(
        "Junction Type",
        [
            "No Junction",
            "T Junction",
            "Crossroad",
            "Roundabout"
        ]
    )

    Area_Type = st.selectbox(
        "Area Type",
        ["Urban", "Rural"]
    )

    Traffic_Level = st.selectbox(
        "Traffic Level",
        ["Low", "Medium", "High"]
    )

    Vehicle_Type = st.selectbox(
        "Vehicle Type",
        ["Car", "Bike", "Bus", "Truck", "Other"]
    )


# Right Column
with col2:

    st.subheader("👤 DRIVER & OTHER DETAILS")

    Speed_Limit = st.number_input(
        "Speed Limit",
        min_value=0,
        value=50,
        step=5
    )

    Number_of_Vehicles = st.number_input(
        "Number Of Vehicles",
        min_value=1,
        value=2
    )

    Number_of_Casualties = st.number_input(
        "Number Of Casualties",
        min_value=0,
        value=1
    )

    Driver_Age = st.number_input(
        "Driver Age",
        min_value=18,
        value=30,
        step=1
    )

    Driver_Experience_Years = st.number_input(
        "Driver Experience (Years)",
        min_value=0,
        value=5,
        step=1
    )

    Visibility_km = st.number_input(
        "Visibility (KM)",
        min_value=0.0,
        value=5.0,
        step=0.5
    )

    Time_of_Day = st.selectbox(
        "Time of Day",
        ["Morning", "Afternoon", "Night"]
    )

    Day_of_Week = st.selectbox(
        "Day of Week",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )


st.divider()


# Prediction Button
if st.button(
    "🔍 PREDICT ACCIDENT SEVERITY",
    use_container_width=True
):

    # Create Input DataFrame
    data = pd.DataFrame([{

        "Weather_Condition": Weather_Condition,
        "Road_Surface_Condition": Road_Surface_Condition,
        "Light_Condition": Light_Condition,
        "Road_Type": Road_Type,
        "Junction_Type": Junction_Type,
        "Area_Type": Area_Type,
        "Traffic_Level": Traffic_Level,
        "Vehicle_Type": Vehicle_Type,

        "Speed_Limit": Speed_Limit,
        "Number_of_Vehicles": Number_of_Vehicles,
        "Number_of_Casualties": Number_of_Casualties,
        "Driver_Age": Driver_Age,
        "Driver_Experience_Years": Driver_Experience_Years,
        "Visibility_km": Visibility_km,
        "Time_of_Day": Time_of_Day,
        "Day_of_Week": Day_of_Week

    }])

    # Convert Categories into Numerical Features
    data = pd.get_dummies(data)

    # Match Training Features
    if not hasattr(model, "feature_names_in_"):
        st.error(
            "Model does not contain feature_names_in_. "
            "Please check the training feature columns."
        )
        st.stop()

    data = data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Predict Severity
    try:
        prediction = model.predict(data)
        severity = str(prediction[0])

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.stop()

    severity_value = severity.strip().lower()

    # High Severity
    if severity_value in ["high", "serious", "fatal"]:

        bg_color = "#fee2e2"
        text_color = "#991b1b"
        border_color = "#ef4444"
        icon = "🚨"

        message = "High accident severity predicted."

        advice = (
            "Reduce speed, maintain a safe following distance, "
            "avoid distractions, and follow all traffic rules."
        )

    # Medium Severity
    elif severity_value in ["medium", "moderate"]:

        bg_color = "#fef3c7"
        text_color = "#92400e"
        border_color = "#f59e0b"
        icon = "⚠️"

        message = "Moderate accident severity predicted."

        advice = (
            "Stay alert, follow speed limits, "
            "avoid sudden braking, and drive cautiously."
        )

    # Low Severity
    elif severity_value in ["low", "slight", "minor"]:

        bg_color = "#dcfce7"
        text_color = "#166534"
        border_color = "#22c55e"
        icon = "✅"

        message = "Low accident severity predicted."

        advice = (
            "Continue following traffic rules, "
            "wear a seat belt or helmet, and stay attentive."
        )

    # Other Predictions
    else:

        bg_color = "#e0e7ff"
        text_color = "#3730a3"
        border_color = "#6366f1"
        icon = "📊"

        message = "Accident severity prediction generated."

        advice = (
            "Check the predicted class against the "
            "labels used during model training."
        )

    # Display Prediction Result
    st.markdown(
        f"""
        <div style="
            background-color: {bg_color};
            color: {text_color};
            padding: 30px;
            border-radius: 15px;
            text-align: center;
            margin-top: 20px;
            border: 2px solid {border_color};
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        ">
            <h2 style="
                color: {text_color} !important;
                font-size: 28px;
                font-weight: bold;
            ">
                {icon} ACCIDENT SEVERITY: {severity.upper()}
            </h2>

            <p style="
                color: {text_color} !important;
                font-size: 18px;
            ">
                {message}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Road Safety Recommendations
    st.markdown("### 🛡️ ROAD SAFETY RECOMMENDATIONS")

    st.info(advice)

    # Model Prediction Confidence
    if hasattr(model, "predict_proba"):

        try:
            probabilities = model.predict_proba(data)[0]

            confidence = max(probabilities) * 100

            st.metric(
                label="📊 Model Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(max(confidence / 100, 0.0), 1.0)
            )

        except Exception:
            st.info("Prediction confidence is unavailable.")

    # Disclaimer
    st.caption(
        "⚠️ This prediction is generated by a machine learning "
        "model and is not a guaranteed assessment of accident "
        "severity. Always follow road safety guidelines."
    )


# Footer
st.divider()

st.markdown(
    """
    <p style="
        text-align: center;
        color: #475569 !important;
        font-size: 14px;
        font-weight: 500;
    ">
        🚗 Road Accident Severity Prediction |
        Powered by Machine Learning & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
