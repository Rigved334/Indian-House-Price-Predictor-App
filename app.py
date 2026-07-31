import streamlit as st
import pandas as pd
import joblib

model = joblib.load(r"C:\Users\Rigved Bhondve\OneDrive\Desktop\ML\Indian House Prices\models\Indian_house_prices.pkl")

st.title("Indian House Price Predictor")

bedrooms = st.number_input("Bedrooms", min_value=1, value=1)
bathrooms = st.number_input("Bathrooms", min_value=1, value=1)
living_area = st.number_input("Living Area", min_value=100, value=100)

lot_area = st.number_input("Lot Area", min_value=100, value=100)

floors = st.number_input("Floors", min_value=1, value=1)

views = st.number_input("Views", min_value=0, value=0)

grade = st.number_input("Grade", min_value=1, value=1)

built_year = st.number_input("Built Year", min_value=1900, value=1900)

renovation_year = st.number_input(
    "Renovation Year",
    min_value=0,
    value=0
)

postal_code = st.number_input(
    "Postal Code",
    value=452010
)

latitude = st.number_input(
    "Latitude",
    value=22.7196
)

longitude = st.number_input(
    "Longitude",
    value=75.8577
)

schools = st.number_input(
    "Schools Nearby",
    min_value=0,
    value=0
)

airport_distance = st.number_input(
    "Distance From Airport",
    min_value=0,
    value=0
)

if st.button("Predict"):
    
    house_age = 2026 - built_year
    years_since_renovation = 2026 - renovation_year

    bedroom_density = bedrooms / living_area
    bathroom_density = bathrooms / living_area

    data = pd.DataFrame([{
        "number of bedrooms": bedrooms,
        "number of bathrooms": bathrooms,
        "living area": living_area,
        "lot area": lot_area,
        "number of floors": floors,
        "waterfront present": 0,
        "number of views": views,
        "condition of the house": 8,
        "grade of the house": grade,
        "Area of the house(excluding basement)": living_area,
        "Area of the basement": 0,
        "Built Year": built_year,
        "Renovation Year": renovation_year,
        "Postal Code": postal_code,
        "Lattitude": latitude,
        "Longitude": longitude,
        "living_area_renov": living_area,
        "lot_area_renov": lot_area,
        "Number of schools nearby": schools,
        "Distance from the airport": airport_distance,
        "house_age": house_age,
        "years_since_renovation": years_since_renovation,
        "bedroom_density": bedroom_density,
        "bathroom_density": bathroom_density
    }])

    prediction = model.predict(data)

    st.success(
        f"Predicted Price: ₹{prediction[0]:,.0f}"
    )