import streamlit as st


def show():
 # Check if the user is an admin
    if not st.session_state.get('admin', False):
        st.error("Access Denied")
        return
    st.title("Admin Dashboard")

    st.write("Welcome to the Admin Dashboard!")
   # Create two columns for layout
    col1, col2 = st.columns(2)
    # Column 1: Plan and Analytics
    with col1:
        st.image("logo1.png", width=150)  # Plan and Analytics logo
        st.write("")  # Blank space
        if st.button("Plan and Analytics"):
            # Navigate to Plan and Analytics page
            st.session_state['current_page'] = 'PlanAndAnalytics'
            st.rerun()


     # Column 2: Churn Prediction
    with col2:
        st.image("logo1.png", width=150)  # Churn Prediction logo
        st.write("")  # Blank space
        if st.button("Churn Prediction"):
            # Navigate to Churn Prediction page
            st.session_state['current_page'] = 'ChurnPrediction'
            st.rerun()


