import streamlit as st
import numpy as np
import pickle
import streamlit.components.v1 as components

# Load models
disorder_model = pickle.load(open('model/disorder_model.pkl', 'rb'))
severity_model = pickle.load(open('model/severity_model.pkl', 'rb'))

# Custom CSS for stylish UI
st.markdown("""
    <style>
    body {
        background-color: #f2f6fc;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-container {
        background-color: white;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        max-width: 800px;
        margin: auto;
    }
    h1.title {
        text-align: center;
        color: #204080;
        margin-bottom: 20px;
    }
    .predict-btn {
        background-color: #204080;
        color: white;
        border-radius: 8px;
        padding: 12px;
        font-size: 16px;
        width: 100%;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Layout start
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🧠 Mental Health Risk Predictor</h1>", unsafe_allow_html=True)

# Input fields
st.subheader("📝 Fill Out This Form")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 10, 80, 25)
    sleep_hours = st.slider("Sleep Hours (per day)", 1, 12, 6)
    stress = st.slider("Stress Level (1 = Low, 10 = High)", 1, 10, 5)

with col2:
    mood_swings = st.selectbox("Mood Swings", ["Yes", "No"])
    past_diagnosis = st.selectbox("Past Diagnosis", ["Yes", "No"])
    distraction = st.selectbox("Distraction Level", ["Low", "Medium", "High"])
    appetite = st.selectbox("Appetite", ["Low", "Medium", "High"])

# Predict button
if st.button("🔮 Predict My Mental Health", key="predict"):
    mood = 1 if mood_swings == "Yes" else 0
    past = 1 if past_diagnosis == "Yes" else 0
    distract = {"Low": 0, "Medium": 1, "High": 2}[distraction]
    food = {"Low": 0, "Medium": 1, "High": 2}[appetite]

    input_data = np.array([[age, sleep_hours, stress, mood, past, distract, food]])

    disorder = disorder_model.predict(input_data)[0]
    severity = severity_model.predict(input_data)[0]

    st.markdown("---")
    st.success(f"🧠 **Detected Disorder:** {disorder}")
    st.info(f"📊 **Severity Level:** {severity.capitalize()}")

    st.markdown("<h4 style='margin-top: 30px;'>📌 Recommended Action:</h4>", unsafe_allow_html=True)

    if severity.lower() == "low":
        st.markdown("- 🧘 Practice daily yoga & mindfulness.")
        st.markdown("- 🛌 Maintain consistent sleep routine.")
    elif severity.lower() == "medium":
        st.markdown("- 🥗 Follow a healthy diet and regular walk.")
        st.markdown("- 💧 Stay hydrated and avoid screen stress.")
    else:
        st.markdown("- 🚨 Please consult a mental health specialist.")
        st.markdown("- 🏥 Suggested: Dr. Naresh – Fortis, Dr. Reena – AIIMS")

# Layout end
st.markdown("</div>", unsafe_allow_html=True)
