import streamlit as st
import pickle
import pandas as pd
import numpy as np
model = pickle.load(open('model.pkl', 'rb'))

def predict_promotion(id, farm_area, temp_obs, wind_direction, dew_temp, pressure_sea_level, precipitation, wind_speed, unix_sec, ingredient_type, farming_company, deidentified_location):

    prediction = model.predict(pd.DataFrame([[id, farm_area, temp_obs, wind_direction, dew_temp, pressure_sea_level, precipitation, wind_speed, unix_sec, ingredient_type, farming_company, deidentified_location]], columns=['id', 'farm_area', 'temp_obs', 'wind_direction', 'dew_temp', 'pressure_sea_level', 'precipitation', 'wind_speed', 'Unix Sec', 'ingredient_type', 'farming_company', 'deidentified_location']))
    return prediction

st.title("Robust yield prediction of various farm processing units:")
html_temp = """ <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Robust yield prediction: Predicting the output of the food processing farms for the next year. </h2>

    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

id = st.number_input("Farm ID")
farm_area = st.number_input("Farm Area")
temp_obs = st.number_input("Temperature Observations")
wind_direction = st.number_input("Wind Direction")
dew_temp = st.number_input("Dew Temperature")
pressure_sea_level = st.number_input("Sea level pressure")
precipitation = st.number_input("Precipitation")
wind_speed = st.number_input("Speed of the wind")
unix_sec = st.number_input("Time")
ingredient_type = st.number_input("Ingredient Type")
farming_company = st.number_input("Farming company")
deidentified_location = st.number_input("Deidentified Compant")

result=""
if st.button("Predict"):
    result=predict_promotion(id, farm_area, temp_obs, wind_direction, dew_temp, pressure_sea_level, precipitation, wind_speed, unix_sec, ingredient_type, farming_company, deidentified_location)
    st.success('The yield is {}'.format(result))



