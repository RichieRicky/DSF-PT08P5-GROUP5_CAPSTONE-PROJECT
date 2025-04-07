import streamlit as st
import joblib
import numpy as np
from datetime import datetime 
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model from the file
model = joblib.load('hiv_prediction_model.pkl')

# label_encoders = joblib.load('label_encoders.pkl')


# Manually created mapping for county names to encoded labels
county_name_to_encoded = {
    'Homa Bay': 0,
    'Kiambu': 1,
    'Kisumu': 2,
    'Migori': 3,
    'Mombasa':4,
    'Nairobi':5,
    'Siaya':6
    # Add all other counties here...
}

county_encoded_to_name = {v: k for k, v in county_name_to_encoded.items()}

head_of_household_mapping = {
    'Father':0,
    'Mother': 1,
    'Husband/Partner': 2,
    'Self': 3,
    'Uncle/Aunt': 4,
    'Grandparents': 5,
    'Sibling': 6,
    'Other/Specify': 7
}

# Streamlit user interface
st.title('HIV Prediction Model')

st.write("Enter the values for the following features:")


# Input fields for each of the important features
year_of_enrollment = st.number_input("Year of Enrollment", min_value= 2000, max_value=2025, value=2009)
age_of_household_head = st.number_input("Age of Household Head", min_value=8, max_value=100, value=30)
age_at_enrollment = st.number_input("Age at Enrollment", min_value=8, max_value=30, value=15)
county = st.selectbox("County", list(county_name_to_encoded.keys()))
no_of_people_in_household = st.number_input("Number of People in Household", min_value=1, max_value=20, value=5)
no_of_children = st.number_input("Number of Children in Household", min_value=0, max_value=20, value=5)
age_at_first_sexual_encounter = st.number_input("Age at First Sexual Encounter", min_value=8, max_value=30, value=15)
no_of_males= st.number_input("Number of Males in Household", min_value=0, max_value=20, value=5)
head_of_household_label = st.selectbox("Head of Household", list(head_of_household_mapping.keys()))
head_of_household_encoded = head_of_household_mapping[head_of_household_label]# Convert label to its encoded value
ever_had_sex = st.selectbox("Ever had Sex", ['Yes', 'No'])
# Convert 'ever_had_sex' to a numeric value: 'Yes' -> 1, 'No' -> 0
ever_had_sex_encoded = 1 if ever_had_sex == 'Yes' else 0


# Create a DataFrame for the input
input_data = pd.DataFrame({
    'year_of_enrollment': [year_of_enrollment],
    'age_of_household_head': [age_of_household_head],
    'age_at_enrollment': [age_at_enrollment],
    'county': [county_name_to_encoded[county]],    
    'no_of_people_in_household': [no_of_people_in_household],
    'no_of_children': [no_of_children],
    'age_at_first_sexual_encounter': [age_at_first_sexual_encounter],
    'no_of_males': [no_of_males],
    'head_of_household': [head_of_household_encoded],
    'ever_had_sex': [ever_had_sex_encoded]  
    
})

# Make prediction

if st.button('Predict'):
    try:
        # Make prediction
        prediction = model.predict(input_data)      
                     
        # Display prediction
        if prediction == 1:
            st.write(f"Prediction: High HIV Risk")
            st.write("Require urgent interventions and clinical investigation.")
        else:
            st.write(f"Prediction: Low HIV Risk")
            st.write("Guidance: Continue regular health check-ups and maintain safe practices.")        
            
    except Exception as e:
        st.error(f"Error in prediction: {e}")