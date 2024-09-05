import streamlit as st
import pandas as pd
from utils.auth import create_user  # Import the function from auth.py
import uuid
from utils.model import predict_plan
def show():
    st.title("Register Page")

    customer_id = str(uuid.uuid4())
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    
    age = st.number_input("Age", min_value=18, max_value=100, value=56)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
    location = st.selectbox("Location", options=["Miami", "New York", "Chicago", "Los Angeles", "Houston"], index=0)
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Widowed"], index=0)
    household_income = st.number_input("Household Income", min_value=0, value=46724)
    family_members = st.number_input("Family Members", min_value=1, value=3)
    education_level = st.selectbox("Education Level", options=["High School", "Associate's", "Bachelor's", "Master's", "Doctorate"], index=0)
    plan_type = st.selectbox("Plan Type", options=["Prepaid", "Postpaid"], index=0)
    subscription_duration = st.number_input("Subscription Duration (months)", min_value=1, value=59)
    monthly_charges = st.number_input("Monthly Charges", min_value=0, value=109)
    data_usage = st.number_input("Data Usage (GB)", min_value=0, value=39)
    voice_minutes = st.number_input("Voice Minutes Usage", min_value=0, value=1864)
    sms_usage = st.number_input("SMS Usage", min_value=0, value=103)
    preferred_comm_channel = st.selectbox("Preferred Communication Channel", options=["Email", "Phone", "SMS"], index=0)
    preferred_comm_time = st.selectbox("Preferred Communication Time", options=["Morning", "Afternoon", "Evening", "Anytime"], index=0)
    preferred_services = st.selectbox("Preferred Services/Features", options=["International Calling", "Data Rollover", "Family Plan", "Unlimited Data"], index=0)
    preferred_payment_method = st.selectbox("Preferred Payment Method", options=["Bank Transfer", "Debit Card", "Online", "Credit Card"], index=0)
    feedback_score = st.number_input("Feedback Score", min_value=0, max_value=10, value=6)
    interactions_with_cs = st.number_input("Interactions with Customer Service", min_value=0, value=0)
    complaints_reported = st.number_input("Complaints Reported", min_value=0, value=0)
    churn_label = st.selectbox("Churn Label", options=["Yes", "No"], index=1)
    
    if st.button("Register"):
        try:
            # Store user credentials in the database
            create_user(customer_id, email, password)
            st.success("Registration Successful!")
            # Add the customer details to the dataset
            add_to_dataset(customer_id, age, gender, location, marital_status, household_income, 
                           family_members, education_level, plan_type, subscription_duration, 
                           monthly_charges, data_usage, voice_minutes, sms_usage, preferred_comm_channel,
                           preferred_comm_time, preferred_services, preferred_payment_method, feedback_score, 
                           interactions_with_cs, complaints_reported, churn_label)
            st.session_state['current_page'] = 'Login'
            predict_plan()
            st.rerun()
        except Exception as e:
            st.error(f"Registration Failed: {e}")

def add_to_dataset(customer_id, age, gender, location, marital_status, household_income, 
                   family_members, education_level, plan_type, subscription_duration, 
                   monthly_charges, data_usage, voice_minutes_usage, sms_usage, preferred_channel,
                   preferred_time, preferred_services, preferred_payment, feedback_score, 
                   interactions, complaints, churn_label):
    # Define the correct column order
    columns = [
        'Customer ID', 'Age', 'Gender', 'Location', 'Marital Status', 'Household Income',
        'Family Members', 'Education Level', 'Plan Type', 'Subscription Duration (months)',
        'Monthly Charges', 'Data Usage (GB)', 'Voice Minutes Usage', 'SMS Usage',
        'Preferred Communication Channel', 'Preferred Communication Time',
        'Preferred Services/Features', 'Preferred Payment Method', 'Feedback Score',
        'Interactions with Customer Service', 'Complaints Reported', 'Churn Label'
    ]
    
    # Read the existing dataset
    df = pd.read_csv('dataset.csv')
    if churn_label=='Yes':
        churn_label=1
    else:
        churn_label=0
    # Create a new DataFrame with the same columns and order
    new_entry = pd.DataFrame([{
        'Customer ID': customer_id,
        'Age': age,
        'Gender': gender,
        'Location': location,
        'Marital Status': marital_status,
        'Household Income': household_income,
        'Family Members': family_members,
        'Education Level': education_level,
        'Plan Type': plan_type,
        'Subscription Duration (months)': subscription_duration,
        'Monthly Charges': monthly_charges,
        'Data Usage (GB)': data_usage,
        'Voice Minutes Usage': voice_minutes_usage,
        'SMS Usage': sms_usage,
        'Preferred Communication Channel': preferred_channel,
        'Preferred Communication Time': preferred_time,
        'Preferred Services/Features': preferred_services,
        'Preferred Payment Method': preferred_payment,
        'Feedback Score': feedback_score,
        'Interactions with Customer Service': interactions,
        'Complaints Reported': complaints,
        'Churn Label': churn_label
    }], columns=columns)
    
    # Concatenate the new entry with the existing DataFrame
    df = pd.concat([df, new_entry], ignore_index=True)
    
    # Write the updated DataFrame back to the CSV file
    df.to_csv('dataset.csv', index=False)
    st.success("Customer details added to the dataset!")
