import streamlit as st

def show():
    st.title("Admin Page")

    # Check if the user is an admin
    if not st.session_state.get('admin', False):
        st.error("Access Denied")
        return

    st.write("Welcome to the Admin Page!")
    # You can add admin-specific functionality here
    # For example, you might want to display a list of all users or manage data
