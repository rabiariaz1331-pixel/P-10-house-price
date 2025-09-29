import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load the trained model
# -------------------------------
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# -------------------------------
# App Title
# -------------------------------
st.title("🏠 House Price Prediction App")

st.write("""
This app predicts **House Prices** based on features:  
- Bedrooms  
- Bathrooms  
- Area (sqft)  
- Location Score  
- Age (years)  
""")

# -------------------------------
# Load dataset (optional preview)
# -------------------------------
try:
    data = pd.read_csv("house_price_dataset_10000.csv")
    if st.checkbox("Show sample dataset"):
        st.write(data.head())
except FileNotFoundError:
    st.warning("⚠️ house_price_dataset_10000.csv not found in the app folder.")

# -------------------------------
# User Inputs
# -------------------------------
st.sidebar.header("Enter House Features")

bedrooms = st.sidebar.number_input("Bedrooms", min_value=1, max_value=20, value=3)
bathrooms = st.sidebar.number_input("Bathrooms", min_value=1, max_value=20, value=2)
area_sqft = st.sidebar.number_input("Area (sqft)", min_value=100, max_value=20000, value=1200)
location_score = st.sidebar.slider("Location Score (1-10)", min_value=1, max_value=10, value=5)
age_years = st.sidebar.number_input("Age (years)", min_value=0, max_value=100, value=10)

# Prepare input for model
features = pd.DataFrame(
    [[bedrooms, bathrooms, area_sqft, location_score, age_years]],
    columns=["Bedrooms", "Bathrooms", "Area_sqft", "Location_score", "Age_years"]
)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Price 💰"):
    try:
        prediction = model.predict(features)
        st.success(f"🏡 Estimated House Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
