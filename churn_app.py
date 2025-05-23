import streamlit as st
import pickle
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
# Load the saved model
with open("churn_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Define the features used in the model
features = ['Overall_Experience_Rating', 'DigitalPlatformUsageScore', 
            'BranchTransactionFrequency', 'ComplaintsRaised', 
            'HighValueCustomer', 'Sum_TotalTransactionAmount']

# Streamlit app section
st.title("Customer Churn Prediction")

st.sidebar.header("Input Features")
# Input fields where user provides feature data
def user_input_features():
    Overall_Exp_Rating = st.sidebar.slider("Overall Experience Rating", 1, 10, 5)
    DigitalPlatformUsageScore = st.sidebar.slider("Digital Platform Usage Score", 0.0, 1.0, 0.5)
    BranchTransactionFrequency = st.sidebar.number_input("Branch Transaction Frequency", min_value=0, value=100)
    ComplaintsRaised = st.sidebar.number_input("Complaints Raised", min_value=0, value=0)
    HighValueCustomer = st.sidebar.selectbox("High Value Customer (1 = Yes, 0 = No)", [1, 0])
    Sum_TotalTransactionAmount = st.sidebar.number_input("Sum of Total Transaction Amount", min_value=0.0, value=100000.0)

    data = {
        'Overall_Experience_Rating': Overall_Exp_Rating,
        'DigitalPlatformUsageScore': DigitalPlatformUsageScore,
        'BranchTransactionFrequency': BranchTransactionFrequency,
        'ComplaintsRaised': ComplaintsRaised,
        'HighValueCustomer': HighValueCustomer,
        'Sum_TotalTransactionAmount': Sum_TotalTransactionAmount
    }
    return pd.DataFrame(data, index=[0])

# Get user input
input_df = user_input_features()

# Show user input
st.subheader("User Input Features")
st.write(input_df)

# Perform prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    # Display results
    st.subheader("Prediction")
    st.write("Churn" if prediction[0] == 1 else "No Churn")

    st.subheader("Prediction Probability")
    st.write(f"Probability of Churn: {prediction_proba[0][1]:.2f}")
    st.write(f"Probability of No Churn: {prediction_proba[0][0]:.2f}")