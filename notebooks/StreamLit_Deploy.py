import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import altair as alt
#streamlit run C:\Users\USER\PycharmProjects\Text\FYP\app.py

class FixedLSTM(LSTM):
    def __init__(self, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(**kwargs)


# Page Configuration
st.set_page_config(
    page_title="AquaCast AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    /* Global Background - Very light cyan/grey tint for a 'clean' feel */
    .stApp {
        background: linear-gradient(to bottom right, #FFFFFF, #F0F4F8);
    }

    /* Headings - Dark Tech Blue */
    h1, h2, h3 {
        color: #023e8a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }

    /* Sidebar styling to match */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }

    /* Custom Card Container */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        border: 1px solid #E1E8ED;
        border-left: 5px solid #00B4D8; /* CYAN ACCENT */
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.08);
    }

    /* Inputs - Cleaner borders */
    .stSelectbox, .stNumberInput, .stSlider {
        margin-bottom: 10px;
    }

    /* The Run Button - TEAL TO CYAN GRADIENT */
    div.stButton > button {
        background: linear-gradient(135deg, #2A9D8F 0%, #00B4D8 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        border-radius: 8px; /* More rectangular/modern than rounded */
        border: none;
        box-shadow: 0 4px 10px rgba(0, 180, 216, 0.3);
        transition: all 0.3s ease;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 180, 216, 0.5);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #FFFFFF;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #F0F4F8;
        color: #00B4D8;
        border-bottom: 2px solid #00B4D8;
    }

</style>
""", unsafe_allow_html=True)

# Location Feature
LOCATION_COORDS = {
    'Albury': (-36.07, 146.91), 'BadgerysCreek': (-33.88, 150.75), 'Cobar': (-31.49, 145.84),
    'CoffsHarbour': (-30.30, 153.11), 'Moree': (-29.46, 149.83), 'Newcastle': (-32.92, 151.78),
    'NorahHead': (-33.28, 151.57), 'NorfolkIsland': (-29.04, 167.95), 'Penrith': (-33.75, 150.70),
    'Richmond': (-33.60, 150.75), 'Sydney': (-33.86, 151.20), 'SydneyAirport': (-33.94, 151.17),
    'WaggaWagga': (-35.12, 147.37), 'Williamtown': (-32.79, 151.84), 'Wollongong': (-34.42, 150.89),
    'Canberra': (-35.28, 149.13), 'Tuggeranong': (-35.42, 149.09), 'MountGinini': (-35.53, 148.77),
    'Ballarat': (-37.56, 143.85), 'Bendigo': (-36.75, 144.27), 'Sale': (-38.10, 147.07),
    'MelbourneAirport': (-37.66, 144.83), 'Melbourne': (-37.81, 144.96), 'Mildura': (-34.20, 142.15),
    'Nhil': (-36.33, 141.65), 'Portland': (-38.36, 141.60), 'Watsonia': (-37.71, 145.08),
    'Dartmoor': (-37.92, 141.27), 'Brisbane': (-27.47, 153.02), 'Cairns': (-16.91, 145.77),
    'GoldCoast': (-28.01, 153.40), 'Townsville': (-19.25, 146.81), 'Adelaide': (-34.92, 138.62),
    'MountGambier': (-37.83, 140.78), 'Nuriootpa': (-34.47, 138.99), 'Woomera': (-31.15, 136.80),
    'Albany': (-35.02, 117.88), 'Witchcliffe': (-34.03, 115.10), 'PearceRAAF': (-31.67, 116.03),
    'PerthAirport': (-31.93, 115.96), 'Perth': (-31.95, 115.86), 'SalmonGums': (-32.98, 121.64),
    'Walpole': (-34.97, 116.73), 'Hobart': (-42.88, 147.32), 'Launceston': (-41.44, 147.14),
    'AliceSprings': (-23.69, 133.87), 'Darwin': (-12.46, 130.84), 'Katherine': (-14.46, 132.26),
    'Uluru': (-25.34, 131.03)
}


# Load Model
@st.cache_resource
def load_all_models():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, 'model')

        # Random Forest
        rf_model = joblib.load(os.path.join(model_dir, 'rf_model_spi.pkl'))
        rf_scaler = joblib.load(os.path.join(model_dir, 'rf_model_scaler.pkl'))
        rf_cols = joblib.load(os.path.join(model_dir, 'rf_model_columns.pkl'))

        # XGBoost
        xgb_model = joblib.load(os.path.join(model_dir, 'xgb_model_spi.pkl'))
        xgb_model.set_params(device="cpu", eval_metric=None)
        if os.path.exists(os.path.join(model_dir, 'xgb_model_scaler.pkl')):
            xgb_scaler = joblib.load(os.path.join(model_dir, 'xgb_model_scaler.pkl'))
        else:
            xgb_scaler = rf_scaler

        # TabNet
        tab_model = joblib.load(os.path.join(model_dir, 'tabnet_model_spi.pkl'))
        if os.path.exists(os.path.join(model_dir, 'tabnet_scaler.pkl')):
            tab_scaler = joblib.load(os.path.join(model_dir, 'tabnet_scaler.pkl'))
        else:
            tab_scaler = rf_scaler

        try:
            lstm_path = os.path.join(model_dir, 'lstm_model_spi.h5')
            lstm_model = load_model(
                lstm_path,
                custom_objects={'LSTM': FixedLSTM},
                compile=False
            )
        except Exception as e:
            st.error(f"Specific LSTM Error: {e}")
            lstm_model = None

        # Load LSTM
        if os.path.exists(os.path.join(model_dir, 'lstm_model_scaler.pkl')):
            lstm_scaler = joblib.load(os.path.join(model_dir, 'lstm_model_scaler.pkl'))
        else:
            lstm_scaler = rf_scaler

        if os.path.exists(os.path.join(model_dir, 'lstm_model_columns.pkl')):
            lstm_cols = joblib.load(os.path.join(model_dir, 'lstm_model_columns.pkl'))
        else:
            lstm_cols = rf_cols

        return (rf_model, rf_scaler, rf_cols,
                xgb_model, xgb_scaler,
                tab_model, tab_scaler,
                lstm_model, lstm_scaler, lstm_cols)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return (None,) * 10


(rf_model, rf_scaler, rf_cols,
 xgb_model, xgb_scaler,
 tab_model, tab_scaler,
 lstm_model, lstm_scaler, lstm_cols) = load_all_models()


# Preprocess Function
def preprocess_input(input_dict, scaler, manual_cols=None):
    # Map Coordinates
    lat, long = LOCATION_COORDS.get(input_dict['Location'], (-33.86, 151.20))
    # Seasonality
    month = input_dict['Month']
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    # Calculated Features
    pressure_intraday = input_dict['Pressure3pm'] - input_dict['Pressure9am']
    humidity_diff = input_dict['Humidity3pm'] - input_dict['Humidity3pm_Lag_3']

    # Construct Dictionary
    data = {
        'MinTemp': [input_dict['MinTemp']],
        'MaxTemp': [input_dict['MaxTemp']],
        'Rainfall': [input_dict['Rainfall']],
        'Evaporation': [input_dict['Evaporation']],
        'Sunshine': [input_dict['Sunshine']],
        'Humidity3pm': [input_dict['Humidity3pm']],
        'Cloud9am': [input_dict['Cloud9am']],
        'Cloud3pm': [input_dict['Cloud3pm']],
        'RainToday': [1 if input_dict['Rainfall'] > 1 else 0],

        #  Location and Time
        'Loc_Lat': [lat],
        'Loc_Long': [long],
        'Month_Sin': [month_sin],
        'Month_Cos': [month_cos],

        #  Derived
        'Pressure_Intraday_Diff': [pressure_intraday],
        'Pressure_Diff': [input_dict['Pressure_Diff']],
        'Humidity_Diff': [humidity_diff],
        'SPI_Diff': [input_dict['SPI_Diff']],
        'SPI_1': [input_dict['SPI_1']],

        # Rolling
        'Rain_Sum_7d': [input_dict['Rain_Sum_7d']],
        'Rain_Sum_14d': [input_dict['Rain_Sum_14d']],
        'Rain_Sum_30d': [input_dict['Rain_Sum_30d']],
        'Temp_Mean_7d': [input_dict['Temp_Mean_7d']],
        'Temp_Mean_14d': [input_dict['Temp_Mean_14d']],
        'Temp_Mean_30d': [input_dict['Temp_Mean_30d']],

        # Lags
        'Humidity3pm_Lag_1': [input_dict['Humidity3pm_Lag_1']],
        'Humidity3pm_Lag_3': [input_dict['Humidity3pm_Lag_3']],
        'Pressure3pm_Lag_1': [input_dict['Pressure3pm_Lag_1']],
        'Pressure3pm_Lag_3': [input_dict['Pressure3pm_Lag_3']],
    }

    df = pd.DataFrame(data)

    if hasattr(scaler, 'feature_names_in_'):
        model_cols = scaler.feature_names_in_
    else:
        model_cols = manual_cols

    # Print mismatch
    missing_cols = set(model_cols) - set(df.columns)
    if missing_cols:
        st.error(f"Missing these columns required by the Scaler: {missing_cols}")
        st.stop()

    df = df.reindex(columns=model_cols, fill_value=0)

    return pd.DataFrame(scaler.transform(df), columns=df.columns)


# SideBar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/414/414927.png", width=50)
    st.title("AquaCast AI")
    st.caption(f"System Status: {'🟢 Online' if rf_model else '🔴 Offline'}")
    st.markdown("### Model Controls")
    sensitivity = st.slider(
        "Risk Sensitivity",
        min_value=1.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Multiplier for Risk classes. 1.0 = Neutral. Higher values make the model more 'paranoid' about Floods/Droughts."
    )
    st.markdown("")
    st.caption(f"Sensitivity Multiplier: **{sensitivity}x**")
    st.markdown("---")
    menu = st.radio("Navigation", ["Dashboard", "System Monitor"])
    st.info("**Tip:** 'Rain Sum 30d' and 'SPI' are the strongest predictors for Flood/Drought.")

# Dashboard
if menu == "Dashboard":
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.title("Hydrological Risk Forecast")
    with col_head2:
        st.write("")

    st.markdown("")

    with st.container():
        st.markdown("#### 📍 Context & Time")
        c_loc, c_month = st.columns([2, 1])
        with c_loc:
            loc_list = sorted(list(LOCATION_COORDS.keys()))
            location = st.selectbox("Select Location", loc_list,
                                    index=loc_list.index('Sydney') if 'Sydney' in loc_list else 0)
        with c_month:
            month = st.selectbox("Select Month", range(1, 13),
                                 format_func=lambda x: f"Month {x}")

    st.markdown("---")

    # Section B: Model Variable
    tab_crit, tab_meteo, tab_adv = st.tabs(["🚨 Critical Indicators", "🌤️ Weather Data", "⚙️ Advanced/Lags"])

    # TAB 1: CRITICAL (The strongest predictors)
    with tab_crit:
        st.info("💡 **Key Insight:** SPI and 30-day Rainfall are the strongest predictors for flood/drought events.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🌧️ Rainfall Accumulation")
            rainfall = st.number_input("Rainfall Today (mm)", value=0.0)
            rain_sum_30d = st.number_input("Total Rain (30 Days)", value=rainfall * 30)

            # Hidden calcs for model consistency
            rain_sum_7d = rainfall * 7
            rain_sum_14d = rainfall * 14
        with c2:
            st.markdown("##### 📉 Standardized Precipitation Index (SPI)")
            spi_input = st.slider("Current SPI Index", -3.0, 3.0, 0.0,
                                  help="Negative = Drought, Positive = Flood")
            spi_diff = st.slider("SPI Trend (3-Day Change)", -1.0, 1.0, 0.0)

    # TAB 2: Meteorological
    with tab_meteo:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🌡️ Temperature & Sun")
            min_temp = st.number_input("Min Temp (°C)", value=15.0)
            max_temp = st.number_input("Max Temp (°C)", value=25.0)
            sunshine = st.number_input("Sunshine (hrs)", value=8.0)
            evaporation = st.number_input("Evaporation (mm)", value=5.0)
        with col2:
            st.markdown("##### 💧 Humidity & Pressure")
            humidity_3pm = st.slider("Humidity 3pm (%)", 0, 100, 50)
            pressure_3pm = st.number_input("Pressure 3pm (hPa)", value=1012.0)
            pressure_9am = st.number_input("Pressure 9am (hPa)", value=1015.0)
            pressure_diff = st.slider("Pressure Trend", -5.0, 5.0, 0.0)

    # TAB 3: Advance
    with tab_adv:
        st.warning("⚠️ These values default to current readings.")
        ac1, ac2 = st.columns(2)
        with ac1:
            cloud_9am = st.slider("Cloud 9am (oktas)", 0, 8, 4)
            cloud_3pm = st.slider("Cloud 3pm (oktas)", 0, 8, 4)
            hum_lag_1 = st.slider("Humidity (Yesterday)", 0, 100, humidity_3pm)
            hum_lag_3 = st.slider("Humidity (3 Days Ago)", 0, 100, humidity_3pm)
        with ac2:
            temp_mean_30d = st.number_input("Mean Temp (30d)", value=max_temp)
            temp_mean_14d = st.number_input("Mean Temp (14d)", value=max_temp)
            temp_mean_7d = st.number_input("Mean Temp (7d)", value=max_temp)
            press_lag_1 = st.number_input("Pressure (Yesterday)", value=pressure_3pm)
            press_lag_3 = st.number_input("Pressure (3 Days Ago)", value=pressure_3pm)

    # Dictionary packing
    user_input_data = {
        'Location': location, 'Month': month,
        'MinTemp': min_temp, 'MaxTemp': max_temp,
        'Rainfall': rainfall, 'Evaporation': evaporation, 'Sunshine': sunshine,
        'Humidity3pm': humidity_3pm, 'Pressure3pm': pressure_3pm, 'Pressure9am': pressure_9am,
        'Cloud9am': cloud_9am, 'Cloud3pm': cloud_3pm,
        'SPI_1': spi_input, 'SPI_Diff': spi_diff,
        'Rain_Sum_30d': rain_sum_30d, 'Rain_Sum_14d': rain_sum_14d, 'Rain_Sum_7d': rain_sum_7d,
        'Pressure_Diff': pressure_diff,

        # New Advanced Mappings
        'Temp_Mean_30d': temp_mean_30d,
        'Temp_Mean_14d': temp_mean_14d,
        'Temp_Mean_7d': temp_mean_7d,
        'Humidity3pm_Lag_1': hum_lag_1,
        'Humidity3pm_Lag_3': hum_lag_3,
        'Pressure3pm_Lag_1': press_lag_1,
        'Pressure3pm_Lag_3': press_lag_3
    }

    # Prediction Action
    st.markdown("###")
    if st.button("Quad Model Analysis", type="primary", use_container_width=True):
        if rf_model and xgb_model and tab_model and lstm_model:
            #  1. PREPROCESS
            X_rf = preprocess_input(user_input_data, rf_scaler, rf_cols)
            X_xgb = preprocess_input(user_input_data, xgb_scaler, rf_cols)
            # TabNet & LSTM use their specific columns
            X_tab = preprocess_input(user_input_data, tab_scaler, lstm_cols)
            X_lstm_df = preprocess_input(user_input_data, lstm_scaler, lstm_cols)

            # 2. PREDICT
            # RF
            prob_rf = rf_model.predict_proba(X_rf)[0]
            pred_rf = np.argmax(prob_rf)

            # XGB
            prob_xgb = xgb_model.predict_proba(X_xgb)[0]
            pred_xgb = np.argmax(prob_xgb)

            # TabNet
            prob_tab = tab_model.predict_proba(X_tab.values)[0]
            pred_tab = np.argmax(prob_tab)

            # LSTM Reshaping
            X_lstm_reshaped = np.reshape(X_lstm_df.values, (1, 1, X_lstm_df.shape[1]))
            prob_lstm = lstm_model.predict(X_lstm_reshaped, verbose=0)[0]
            pred_lstm = np.argmax(prob_lstm)

            #  3. LABELS
            labels_map = {0: "Normal", 1: "Drought", 2: "Flood"}
            label_rf = labels_map.get(pred_rf, "Unknown")
            label_xgb = labels_map.get(pred_xgb, "Unknown")
            label_tab = labels_map.get(pred_tab, "Unknown")
            label_lstm = labels_map.get(pred_lstm, "Unknown")

            #  4. RESULTS DISPLAY (2x2 Grid)
            st.markdown("### Ensemble Results")

            # Row 1
            r1c1, r1c2 = st.columns(2)
            # Row 2
            r2c1, r2c2 = st.columns(2)


            # Function to render card
            def render_card(col, name, label, prob, color):
                with col:
                    st.markdown(f"""
                        <div style="background-color: white; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <h4 style="margin:0;">{name}</h4>
                            <h2 style="color: {'#00b894' if label == 'Normal' else '#d63031'}; margin: 5px 0;">
                                {label.upper()}
                            </h2>
                            <p style="margin:0; font-size: 0.9em;">Confidence: <b>{max(prob):.1%}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    with st.expander(f"View {name} Details"):
                        st.progress(float(prob[0]), text=f"Normal: {prob[0]:.2f}")
                        st.progress(float(prob[1]), text=f"Drought: {prob[1]:.2f}")
                        st.progress(float(prob[2]), text=f"Flood: {prob[2]:.2f}")

            # Render
            render_card(r1c1, "Random Forest", label_rf, prob_rf, "#00b894")
            render_card(r1c2, "XGBoost", label_xgb, prob_xgb, "#0984e3")
            render_card(r2c1, "TabNet(Deep Learning)", label_tab, prob_tab, "#6c5ce7")
            render_card(r2c2, "LSTM (Deep Learning)", label_lstm, prob_lstm, "#e17055")

            # Consensus Logic
            st.markdown("---")
            st.subheader("Consensus Intelligence")

            # 1. Calculate Raw Averages
            all_probs = np.array([prob_rf, prob_xgb, prob_tab, prob_lstm])
            avg_probs = np.mean(all_probs, axis=0)

            # Apply Sensitivity Logic
            adjusted_probs = avg_probs.copy()

            # Multiply Drought and Flood by the slider value
            adjusted_probs[1] *= sensitivity
            adjusted_probs[2] *= sensitivity

            # Determine Winner based on Adjusted Probabilities
            final_pred_idx = np.argmax(adjusted_probs)
            final_verdict = labels_map.get(final_pred_idx, "Unknown")
            raw_pred_idx = np.argmax(avg_probs)
            raw_verdict = labels_map.get(raw_pred_idx, "Unknown")

            # Display Logic
            col_res, col_metric = st.columns([3, 1])

            with col_res:
                # Case A: Sensitivity has casue a changed in the result
                if raw_verdict != final_verdict:
                    st.warning(
                        f"⚠️ **Sensitivity Triggered:** Raw consensus was **{raw_verdict}**, but sensitivity settings flagged **{final_verdict}**.")
                    st.markdown(f"The system has elevated the risk status to **{final_verdict}** due to  safety threshold.")

                # Case B: Consensus logic
                else:
                    votes = [label_rf, label_xgb, label_tab, label_lstm]
                    agreement_count = votes.count(final_verdict)

                    if agreement_count == 4:
                        st.success(f"**Unanimous Decision:** All models agree on **{final_verdict}**.")
                    elif agreement_count == 3:
                        st.info(f"**Strong Consensus:** Majority predict **{final_verdict}**.")
                    else:
                        st.warning(f"**Split Decision:** Models are diverging, but aggregate leans towards **{final_verdict}**.")

            with col_metric:
                final_conf = adjusted_probs[final_pred_idx]
                # Cap it at 100%
                final_conf = min(final_conf, 1.0)

                st.metric(
                    label="Risk Confidence",
                    value=f"{final_conf:.1%}",
                    delta="Boosted" if sensitivity > 1.0 and final_pred_idx != 0 else "Raw"
                )

            # Visualization
            st.caption("Aggregated Probability Distribution")

            # Visualization Grid
            # Normal
            st.markdown("**Normal**")
            st.progress(float(avg_probs[0]), text=f"Raw Probability: {avg_probs[0]:.1%}")

            # Drought
            st.markdown("**Drought**")
            d_col1, d_col2 = st.columns([5, 1])
            with d_col1:
                st.progress(float(avg_probs[1]), text=f"Raw Probability: {avg_probs[1]:.1%}")
            with d_col2:
                if sensitivity > 1.0:
                    st.caption(f"Adj: **{min(adjusted_probs[1], 1.0):.1%}**")

            # FLOOD
            st.markdown("**Flood**")
            f_col1, f_col2 = st.columns([5, 1])
            with f_col1:
                st.progress(float(avg_probs[2]), text=f"Raw Probability: {avg_probs[2]:.1%}")
            with f_col2:
                if sensitivity > 1.0:
                    st.caption(f"Adj: **{min(adjusted_probs[2], 1.0):.1%}**")

            if sensitivity > 1.0:
                st.info(f"ℹ️ Probabilities for Flood and Drought are being multiplied by **{sensitivity}x** for decision making.")

elif menu == "System Monitor":
    st.title("⚙️ System Monitor & Architecture")
    st.markdown("Technical details regarding the Quad-Model ensemble performance and configuration.")

    # 1. MODEL ARCHITECTURE
    st.subheader("Model Architecture")
    with st.expander("View Ensemble Logic", expanded=True):
        st.markdown("""
        The system utilizes a **Weighted Soft-Voting Ensemble** combining four distinct architectures:

        * **Random Forest (RF):** Handles non-linear relationships and high-dimensional interactions.
        * **XGBoost (Gradient Boosting):** Optimized for speed and performance on structured data.
        * **TabNet (Attention-based):** A deep learning model that selects features at each decision step (interpretable AI).
        * **LSTM (Recurrent Neural Network):** Captures temporal dependencies and sequential patterns (Time-Series).
        """)

        #  Diagram using Streamlit Graphviz
        st.graphviz_chart("""
            digraph {
                rankdir=LR;
                node [shape=box, style=filled, fillcolor="#E1E8ED"];
                Input [label="Input Features\n(Met + Time)", shape=ellipse, fillcolor="#00B4D8", fontcolor=white];
                RF [label="Random Forest"];
                XGB [label="XGBoost"];
                Tab [label="TabNet"];
                LSTM [label="LSTM (Seq)"];
                Vote [label="Voting", shape=diamond, fillcolor="#2A9D8F", fontcolor=white];
                Output [label="Prediction\n(Normal/Flood/Drought)", shape=ellipse, fillcolor="#00B4D8", fontcolor=white];

                Input -> RF;
                Input -> XGB;
                Input -> Tab;
                Input -> LSTM;
                RF -> Vote;
                XGB -> Vote;
                Tab -> Vote;
                LSTM -> Vote;
                Vote -> Output;
            }
        """)

    # Feature Importance
    st.subheader("RF Feature Importance")
    st.info("Top predictors extracted directly from the trained Random Forest model.")

    if rf_model is not None and rf_cols is not None:
        try:
            # Get the raw scores
            importances = rf_model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'Feature': rf_cols,
                'Importance': importances
            })

            # Sort Data
            feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
            # Take the top 10
            top_features = feature_imp_df.head(10)

            # Display with Altair
            c = alt.Chart(top_features).mark_bar(color="#00B4D8").encode(
                x=alt.X('Feature', sort='-y', title="Feature Name"),
                y=alt.Y('Importance', title="Importance Score"),
                tooltip=['Feature', 'Importance']
            ).properties(
                height=400
            )

            st.altair_chart(c, use_container_width=True)

            # Optional: Show the full list in a table
            with st.expander("View Full Sorted List"):
                st.dataframe(feature_imp_df, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not extract feature importance: {e}")
    else:
        st.error("Model not loaded.")

    # System Health Check
    st.subheader("System Diagnostics")

    # Calculate Model Count
    models_list = [rf_model, xgb_model, tab_model, lstm_model]
    active_count = sum(1 for m in models_list if m is not None)

    # Check Scalers
    scalers_list = [rf_scaler, xgb_scaler, tab_scaler, lstm_scaler]
    scalers_ok = all(s is not None for s in scalers_list)
    scaler_status = "Synced" if scalers_ok else "Missing"
    scaler_delta = "Optimal" if scalers_ok else "Critical"

    # Get Feature Count
    feature_count = len(rf_cols) if rf_cols is not None else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        # RShows actual number of loaded models
        st.metric(
            label="Active Models",
            value=f"{active_count} / 4",
            delta="Fully Online" if active_count == 4 else "Degraded"
        )
    with c2:
        # Checks if scalers exist
        st.metric(
            label="Scaler Status",
            value=scaler_status,
            delta=scaler_delta
        )
    with c3:
        # Shows actual input dimension expected by the model
        st.metric(
            label="Input Features",
            value=f"{feature_count}",
            help="Number of input variables (columns) the model expects"
        )

    # Data Dictionary
    with st.expander("Variable Data Dictionary"):
        dict_df = pd.DataFrame([
            {"Variable": "Location", "Description": "Geographic location of the weather station (Latitude/Longitude)."},
            {"Variable": "Month", "Description": "Month of observation (Processed into Cyclical Sin/Cos features)."},
            {"Variable": "MinTemp", "Description": "Minimum temperature in degrees Celsius (°C)."},
            {"Variable": "MaxTemp", "Description": "Maximum temperature in degrees Celsius (°C)."},
            {"Variable": "Rainfall", "Description": "Rainfall recorded for the day in millimeters (mm)."},
            {"Variable": "Evaporation", "Description": "Class A pan evaporation in millimeters (mm)."},
            {"Variable": "Sunshine", "Description": "Number of hours of bright sunshine."},
            {"Variable": "Humidity3pm", "Description": "Relative humidity percentage at 3pm (%)."},
            {"Variable": "Pressure9am", "Description": "Atmospheric pressure reduced to mean sea level at 9am (hPa)."},
            {"Variable": "Pressure3pm", "Description": "Atmospheric pressure reduced to mean sea level at 3pm (hPa)."},
            {"Variable": "Cloud9am", "Description": "Fraction of sky obscured by cloud at 9am."},
            {"Variable": "Cloud3pm", "Description": "Fraction of sky obscured by cloud at 3pm."},
            {"Variable": "SPI_1", "Description": "Standardized Precipitation Index (1-month). Primary drought/flood indicator."},
            {"Variable": "SPI_Diff", "Description": "Trend/Change in SPI over a 3-day window."},
            {"Variable": "Rain_Sum_7d", "Description": "Cumulative rainfall over the past 7 days."},
            {"Variable": "Rain_Sum_14d", "Description": "Cumulative rainfall over the past 14 days."},
            {"Variable": "Rain_Sum_30d", "Description": "Cumulative rainfall over the past 30 days (High predictor)."},
            {"Variable": "Rain_Today", "Description": "Whether It Rain Today"},
            {"Variable": "Temp_Mean_7d", "Description": "Average Max Temperature over the past 7 days."},
            {"Variable": "Temp_Mean_14d", "Description": "Average Max Temperature over the past 14 days."},
            {"Variable": "Temp_Mean_30d", "Description": "Average Max Temperature over the past 30 days."},
            {"Variable": "Pressure_Diff", "Description": "Calculated trend in pressure variations."},
            {"Variable": "Pressure_Intraday", "Description": "Difference between 3pm and 9am pressure (Calculated automatically)."},
            {"Variable": "Humidity_Diff", "Description": "Humidity Change (3 Day Ago"},
            {"Variable": "Humidity3pm_Lag_1", "Description": "Historical Humidity (3pm) from 1 day ago."},
            {"Variable": "Humidity3pm_Lag_3", "Description": "Historical Humidity (3pm) from 3 days ago."},
            {"Variable": "Pressure3pm_Lag_1", "Description": "Historical Pressure (3pm) from 1 day ago."},
            {"Variable": "Pressure3pm_Lag_3", "Description": "Historical Pressure (3pm) from 3 days ago."}
        ])
        st.table(dict_df)

