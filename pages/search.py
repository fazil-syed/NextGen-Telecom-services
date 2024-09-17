import streamlit as st
import pandas as pd

def show():
    st.title("Plan Page")

   
    customerid = st.session_state.get('customer_id')

   
    if customerid:
        customerid = customerid[0]
       
        df_predictions = pd.read_csv('plans_dataset.csv')
        

        
        result = df_predictions[df_predictions['Customer ID'] == customerid]
        

        if not result.empty:
        
            st.write("**Customer Details:**")
            row_prediction = result.iloc[0]
            st.write(f"**Best Service Recommended (Topsis):** {row_prediction['BestServiceNameTopsis']}")
            st.write(f"**Best Service Recommended (IFS):** {row_prediction['BestServiceNameCosine']}")
            st.write(f"**Best Service Recommended (cosine):** {row_prediction['BestServiceNameIFS']}")
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