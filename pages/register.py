import streamlit as st
import pandas as pd
from utils.auth import create_user 
import uuid
from utils.model import predict_plan, recommend_plan,recommend_plan_topsis,recommend_plan_ifs
def show():
    st.title("Register Page")

    customer_id = str(uuid.uuid4())
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    
    age = st.number_input("Age", min_value=18, max_value=100, value=56)
    gender = st.selectbox("Gender", options=["Male", "Female", "Other"], index=0)
    location = st.selectbox("Location", options=["Miami", "New York", "Chicago", "Los Angeles", "Houston"], index=0)
    education_level = st.selectbox("Education Level", options=["High School", "Associate's", "Bachelor's", "Master's", "Doctorate"], index=0)
    
    
    if st.button("Register"):
        try:
           
            create_user(customer_id, email, password)
            st.success("Registration Successful!")
           
            planTop=recommend_plan_topsis(age,gender,location,education_level)
            planCos=recommend_plan(age,gender,location,education_level)
            planIfs =recommend_plan_ifs(age,gender,location,education_level)
            add_to_dataset(customer_id, age, gender, location, education_level,planTop,planCos,planIfs)
            st.session_state['current_page'] = 'Login'
            predict_plan()
            st.rerun()
        except Exception as e:
            st.error(f"Registration Failed: {e}")

def add_to_dataset(customer_id, age, gender, location, 
                  education_level,planTop,planCos,planIfs):
   
    columns = [
        'Customer ID', 'Age', 'Gender', 'Location',  'Education Level', 'BestServiceNameTopsis','BestServiceNameCosine','BestServiceNameIFS'
    ]
    
    
    df = pd.read_csv('plans_dataset.csv')
  
    new_entry = pd.DataFrame([{
        'Customer ID': customer_id,
        'Age': age,
        'Gender': gender,
        'Location': location,
        'Education Level': education_level,
        'BestServiceNameTopsis':planTop,
        'BestServiceNameCosine':planCos,
        'BestServiceNameIFS':planIfs
        
    }], columns=columns)
   
    df = pd.concat([df, new_entry], ignore_index=True)
    
    
    df.to_csv('plans_dataset.csv', index=False)
    st.success("Customer details added to the dataset!")
