import streamlit as st
import numpy as np
import joblib

# Load the model
model = joblib.load('hiv_prediction_model.pkl')

# Streamlit app UI to get user input
st.title("HIV Prediction System")


county = st.selectbox("County", ["Nairobi", "Kiambu", "Mombasa", "Siaya", "Kisumu","Migori","Homa Bay"]) 
head_of_household
age_of_head = st.number_input("Age of Household Head", min_value=9)
father_alive = st.selectbox("Father Alive?", ["Yes", "No"])
mother_alive = st.selectbox("Mother Alive?", ["Yes", "No"])
are_both_parents_alive = 
no_of_people_in_household
no_of_females
no_of_males
no_of_adults
no_of_children
ever_missed_full_day_food_in_4wks
no_of_days_missed_food_in_4wks
currently_in_school
current_school_level
current_income_source
last_test_result
ever_had_sex
age_at_first_sexual_encounter
has_sexual_partner
age_of_last_partner
used_condom_with_last_partner
exit_status
exit_reason
bio_medical
social_protection
behavioral
post_gbv_care
other_interventions
age_at_enrollment



# Convert 'Yes'/'No' to 1/0
father_alive = 1 if father_alive == 'Yes' else 0
mother_alive = 1 if mother_alive == 'Yes' else 0

# Create the feature array
features = np.array([age_of_head, 
                     county, 
                     father_alive, mother_alive,
                     are_both_parents_alive,
                     head_of_household,
                     no_of_people_in_household,
                     no_of_females,
                     no_of_males,
                     no_of_adults,
                     no_of_children,
                     ever_missed_full_day_food_in_4wks,
                     no_of_days_missed_food_in_4wks,
                     currently_in_school,
                     current_school_level,
                     current_income_source,
                     last_test_result,
                     ever_had_sex,
                     age_at_first_sexual_encounter,
                     has_sexual_partner,
                     age_of_last_partner,
                     used_condom_with_last_partner,
                     exit_status,
                     exit_reason,
                     bio_medical,
                     social_protection,
                     behavioral,
                     post_gbv_care,
                     other_interventions,
                     age_at_enrollment
])  # Add more features as needed

# Prediction
if st.button("Predict HIV Status"):
    prediction = model.predict(features.reshape(1, -1))
    if prediction[0] == 1:
        st.success("HIV Positive Prediction")
    else:
        st.success("HIV Negative Prediction")