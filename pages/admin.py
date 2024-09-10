import streamlit as st


def show():

    if not st.session_state.get('admin', False):
        st.error("Access Denied")
        return
    st.title("Admin Dashboard")

    st.write("Welcome to the Admin Dashboard!")
 
    col1, col2 = st.columns(2)
    
    with col1:
        st.image("logo1.png", width=150)  
        st.write("")  
        if st.button("Plan and Analytics"):
           
            st.session_state['current_page'] = 'PlanAndAnalytics'
            st.rerun()


    with col2:
        st.image("logo1.png", width=150)  
        st.write("") 
        if st.button("Churn Prediction"):
         
            st.session_state['current_page'] = 'ChurnPrediction'
            st.rerun()


