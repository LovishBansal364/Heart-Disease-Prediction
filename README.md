# 🫀 Heart Disease Prediction App

This is a **machine learning-powered web app** built with **Streamlit** that predicts the risk of heart disease.  
It uses a trained **KNN model** along with preprocessing (scaling + feature encoding) to provide predictions based on user inputs.

---

## 🚀 Features
- Interactive and user-friendly UI (sliders & dropdowns)  
- Real-time predictions (High Risk / Low Risk)  
- Probability-based risk score  
- Visualization with an interactive gauge chart  
- Built with **Python, scikit-learn, Streamlit, Plotly, and Joblib**  

---

## 📂 Project Structure
├── app.py # Main Streamlit app
├── KNN_heart.pkl # Trained KNN model
├── scaler.pkl # Scaler used during training
├── columns.pkl # Feature columns used for encoding
├── requirements.txt # Dependencies
└── README.md # Project documentation

## 🌐 Deployment

You can easily deploy this app using Streamlit Community Cloud:

Push this repo to GitHub.

Go to Streamlit Cloud
 → New App → Select your repo.

Make sure requirements.txt includes:
streamlit
joblib
numpy
pandas
plotly
scikit-learn

## 🎉 Deploy 

📊 Example Output

✅ No significant risk of Heart Disease

⚠️ High chance of Heart Disease

Along with a risk probability gauge chart.

