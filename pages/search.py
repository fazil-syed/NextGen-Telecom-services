import streamlit as st
import pandas as pd

def show():
    st.title("Plan Page")

    # Retrieve customer ID from session state
    customerid = st.session_state.get('customer_id')

    # Check if a customer ID is available in session state
    if customerid:
        customerid = customerid[0]
        # Load the datasets
        df_predictions = pd.read_csv('plans_dataset.csv')
        

        # Filter the DataFrames to get the row with the matching CustomerID
        result = df_predictions[df_predictions['Customer ID'] == customerid]
        

        if not result.empty:
            # Display the basic details
            st.write("**Customer Details:**")
            row_prediction = result.iloc[0]
            st.write(f"**Best Service Recommended:** {row_prediction['BestServiceName']}")
            st.write(f"**Age:** {result.iloc[0]['Age']}")
            st.write(f"**Gender:** {result.iloc[0]['Gender']}")
            st.write(f"**Location:** {result.iloc[0]['Location']}")
            st.write(f"**Education Level:** {result.iloc[0]['Education Level']}")
           

        else:
            st.write("No customer found with the provided ID.")

    else:
        st.error("No customer ID found in session state.")

    if st.button("Back to Home"):
      
       
        st.session_state['current_page'] = 'Home'
                
        st.rerun()