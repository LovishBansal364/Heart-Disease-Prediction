import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------------
# Load Pickle Files
# ------------------------
with open("KNN_heart.pkl", "rb") as f:
    model = joblib.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = joblib.load(f)

with open("columns.pkl", "rb") as f:
    columns = joblib.load(f)   # list of column names in the same order used for training


# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Enter the details below to check the risk of heart disease.")

# Age as slider
age = st.slider("Age", min_value=1, max_value=120, value=40)

# Other numeric inputs
restingBP = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=200)
fastingBS = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])

# Now sliders for MaxHR and Oldpeak
maxHR = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
oldpeak = st.slider("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)

# Dropdown categorical inputs
sex = st.selectbox("Sex", ["M", "F"])
cp_type = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA"])
restingECG = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
st_slope = st.selectbox("ST Slope", ["Flat", "Up", "Down"])
exercise_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])


# ------------------------
# Process Input into Feature Vector
# ------------------------
# Initialize all columns as 0
input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Fill numeric
input_data.at[0, "Age"] = age
input_data.at[0, "RestingBP"] = restingBP
input_data.at[0, "Cholesterol"] = cholesterol
input_data.at[0, "FastingBS"] = fastingBS
input_data.at[0, "MaxHR"] = maxHR
input_data.at[0, "Oldpeak"] = oldpeak

# Fill categorical (one-hot encoded style)
if sex == "M":
    input_data.at[0, "Sex_M"] = 1

if cp_type in ["ATA", "NAP", "TA"]:
    input_data.at[0, f"ChestPainType_{cp_type}"] = 1

if restingECG in ["Normal", "ST"]:
    input_data.at[0, f"RestingECG_{restingECG}"] = 1

if st_slope in ["Flat", "Up"]:
    input_data.at[0, f"ST_Slope_{st_slope}"] = 1

if exercise_angina == "Yes":
    input_data.at[0, "ExerciseAngina_Y"] = 1

# ------------------------
# Scale the data
# ------------------------
input_scaled = scaler.transform(input_data)

# ------------------------
# Prediction
# ------------------------
# if st.button("🔍 Predict"):
#     prediction = model.predict(input_scaled)[0]
#     if prediction == 1:
#         st.error("⚠️ High chance of Heart Disease.")
#     else:
#         st.success("✅ No significant risk of Heart Disease.")



# ------------------------
# Prediction
# ------------------------
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]  # probability scores
    risk_percent = round(proba[1] * 100, 2)  # % chance of disease

    if prediction == 1:
        st.error("⚠️ High chance of Heart Disease.")
    else:
        st.success("✅ No significant risk of Heart Disease.")

    # --- Add Gauge Chart ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={'text': "Heart Disease Risk (%)"},
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

