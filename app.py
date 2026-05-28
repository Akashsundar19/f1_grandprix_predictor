import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="F1 Grand Prix Winner Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================
# LOAD DATASET
# =========================================
@st.cache_data
def load_data():
    return pd.read_csv("f1_data.csv")

data = load_data()

# =========================================
# LOAD MODEL & ENCODERS
# =========================================
@st.cache_resource
def load_models():
    model = pickle.load(open("f1_model.pkl", "rb"))
    driver_encoder = pickle.load(open("driver_encoder.pkl", "rb"))
    team_encoder = pickle.load(open("team_encoder.pkl", "rb"))
    circuit_encoder = pickle.load(open("circuit_encoder.pkl", "rb"))
    return model, driver_encoder, team_encoder, circuit_encoder

model, driver_encoder, team_encoder, circuit_encoder = load_models()

# =========================================
# CUSTOM F1 THEME CSS
# =========================================
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
    }
    
    .title-container {
        text-align: center;
        padding: 20px 0px 5px 0px;
    }
    .main-title {
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -1px;
        color: #ffffff;
        text-transform: uppercase;
        margin-bottom: 0px;
    }
    .red-accent {
        color: #e10600;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #8b949e;
        margin-bottom: 35px;
    }

    .metric-card {
        background: #161b22;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #e10600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        text-align: left;
    }
    .metric-title {
        color: #8b949e;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        color: #f0f6fc;
        font-size: 32px;
        font-weight: bold;
        margin-top: 5px;
    }

    .stButton > button {
        width: 100%;
        background-color: #e10600 !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ff1e18 !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(225, 6, 0, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# =========================================
# HEADER BLOCK
# =========================================
st.markdown("""
    <div class="title-container">
        <h1 class="main-title">🏁 Formula 1 <span class="red-accent">Winner Predictor</span></h1>
    </div>
""", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning analytics dashboard fueled by real historical Grand Prix data</div>', unsafe_allow_html=True)

# =========================================
# TOP LEVEL KPI METRICS
# =========================================
total_races = int(data['Circuit'].nunique())
total_drivers = int(data['Driver'].nunique())
total_teams = int(data['Team'].nunique())

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Circuits Analyzed</div><div class="metric-value">{total_races}</div></div>', unsafe_allow_html=True)
with m_col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Active Drivers</div><div class="metric-value">{total_drivers}</div></div>', unsafe_allow_html=True)
with m_col3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">Constructors</div><div class="metric-value">{total_teams}</div></div>', unsafe_allow_html=True)

st.write("")
st.write("---")

# =========================================
# PREDICTION ENGINE (INPUTS)
# =========================================
st.subheader("🏁 Grand Prix Prediction Engine")

# Filter alignment with safe matching classes
available_drivers = sorted([d for d in driver_encoder.classes_ if d in data['Driver'].unique()])
available_circuits = sorted([c for c in circuit_encoder.classes_ if c in data['Circuit'].unique()])

col1, col2 = st.columns(2)

with col1:
    selected_driver = st.selectbox("Select Driver", available_drivers)
    
    # Isolate valid teams for the chosen driver to prevent encoder mismatch crashes
    valid_teams_for_driver = data[data['Driver'] == selected_driver]['Team'].unique()
    available_teams = sorted([t for t in team_encoder.classes_ if t in valid_teams_for_driver])
    
    if not available_teams:
        available_teams = sorted([t for t in team_encoder.classes_ if t in data['Team'].unique()])

    selected_team = st.selectbox("Select Constructor", available_teams)

with col2:
    grid_position = st.slider("Grid Position", 1, 20, 1)
    selected_circuit = st.selectbox("Select Circuit", available_circuits)

st.write("")

# =========================================
# SIMULATION OUTPUT
# =========================================
if st.button("🚀 Run Prediction Simulation"):
    try:
        # Encode values safely
        driver_encoded = driver_encoder.transform([selected_driver])[0]
        team_encoded = team_encoder.transform([selected_team])[0]
        circuit_encoded = circuit_encoder.transform([selected_circuit])[0]

        # Array structure mapped to matching model requirements
        input_data = np.array([[
            driver_encoded,
            team_encoded,
            grid_position,
            circuit_encoded
        ]])

        # Inference Processing
        probability_array = model.predict_proba(input_data)[0]
        probability = probability_array[1] if len(probability_array) > 1 else probability_array[0]
        probability_percent = round(probability * 100, 2)

        # Output Card Visuals
        st.markdown(f"""
            <div style="
                background: #161b22;
                padding: 35px;
                border-radius: 16px;
                border: 2px solid #e10600;
                text-align: center;
                margin-top: 25px;
                margin-bottom: 15px;
            ">
                <h3 style="color: #8b949e; font-size: 20px; margin: 0; text-transform: uppercase;">🏁 Predicted Race Winner</h3>
                <h1 style="color: #ffffff; font-size: 56px; margin: 15px 0;">{selected_driver}</h1>
                <h2 style="color: #e10600; font-size: 28px; margin: 0; font-weight: bold;">Win Probability: {probability_percent}%</h2>
            </div>
        """, unsafe_allow_html=True)

        st.progress(float(probability))

        if probability_percent > 70:
            st.success("🔥 Extremely strong winning probability! Highly favored victory criteria.")
        elif probability_percent > 40:
            st.warning("⚠️ Moderate winning probability. Podiums likely, strategic operational variations expected.")
        else:
            st.error("❌ Low winning probability based on historical telemetry layout configurations.")

    except Exception as e:
        st.error(f"Prediction Stream interrupted due to categorical evaluation error: {str(e)}")

# =========================================
# ANALYTICS DASHBOARD HUB
# =========================================
st.write("")
st.write("---")
st.subheader("📊 Formula 1 Analytics Dashboard")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    # 1. Driver Wins
    winner_data = data[data['Winner'] == 1]
    if not winner_data.empty:
        wins = winner_data['Driver'].value_counts().reset_index()
        wins.columns = ['Driver', 'Wins']
        
        fig1 = px.bar(
            wins.head(12),
            x='Driver',
            y='Wins',
            title='🏆 Race Wins by Driver (Top 12)',
            template='plotly_dark',
            color_discrete_sequence=['#e10600']
        )
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No conversion metrics found for the active 'Winner' data configuration.")

with chart_col2:
    # 2. Team Grid Position
    grid_col = 'GridPosition' if 'GridPosition' in data.columns else 'grid_position'
    if grid_col in data.columns:
        team_grid = data.groupby('Team')[grid_col].mean().reset_index().sort_values(by=grid_col)
        
        fig2 = px.line(
            team_grid,
            x='Team',
            y=grid_col,
            markers=True,
            title='📈 Average Grid Position by Team (Lower is Better)',
            template='plotly_dark',
            color_discrete_sequence=['#e10600']
        )
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

# 3. Full-width Circuit breakdown
st.write("")
circuit_counts = data['Circuit'].value_counts().reset_index()
circuit_counts.columns = ['Circuit', 'Entries']

fig3 = px.bar(
    circuit_counts,
    x='Circuit',
    y='Entries',
    color='Entries',
    title='🌍 Circuit Sample Distribution footprint',
    template='plotly_dark',
    color_continuous_scale='Reds'
)
fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig3, use_container_width=True)

# =========================================
# DATASET PREVIEW
# =========================================
st.write("")
st.subheader("📋 Formula 1 Dataset Preview")

st.dataframe(
    data.head(20),
    use_container_width=True
)

# =========================================
# FOOTER
# =========================================
st.write("")
st.write("---")
st.markdown('<div style="text-align: center; color: #8b949e; padding: 10px; font-size:13px;">Built using FastF1 • Scikit-learn • Plotly • Streamlit Core Framework</div>', unsafe_allow_html=True)