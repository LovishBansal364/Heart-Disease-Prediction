# 🫀 Heart Disease Prediction App

This is a **machine learning-powered web app** built with **Streamlit** that predicts the risk of heart disease.  
The backend is powered by a **K-Nearest Neighbors (KNN) model** that I trained **from scratch**, including all preprocessing (feature encoding, scaling, and model training).  

---

## 🚀 Features

- Interactive and user-friendly UI (sliders & dropdowns)  
- Real-time predictions (High Risk / Low Risk)  
- Probability-based risk score  
- Visualization with an interactive gauge chart  
- **Custom ML pipeline built from scratch (scikit-learn, preprocessing, model training)**  
- Built with **Python, scikit-learn, Streamlit, Plotly, and Joblib**  

---

## 📂 Project Structure

├── app.py - Main Streamlit app
├── KNN_heart.pkl - Trained KNN model
├── scaler.pkl - Scaler used during training
├── columns.pkl - Feature columns used for encoding
├── requirements.txt - Dependencies
└── README.md - Project documentation

## 🌐 Deployment

You can easily deploy this app using Streamlit Community Cloud:
1. Push this repo to GitHub.
2. Go to Streamlit Cloud
 → New App → Select your repo.
3. Make sure requirements.txt includes:
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

