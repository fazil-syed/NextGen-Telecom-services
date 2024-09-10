import streamlit as st
from utils.auth import authenticate_user, get_customer_id_by_email

def show():
    st.title("Login Page")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    is_admin = st.checkbox("Login as Admin")

    if st.button("Login"):
        if is_admin:
          
            if email == 'saujanya@gmail.com' and password == 'saujanya':
                st.session_state['logged_in'] = True
                st.session_state['admin'] = True
                st.session_state['current_page'] = 'Admin'
                st.success("Admin Login Successful!")
                st.rerun()
            else:
                st.error("Invalid admin email or password")
        else:
         
            customer_id = authenticate_user(email, password)
            if customer_id:
                st.session_state['logged_in'] = True
                st.session_state['admin'] = False
                st.session_state['customer_id'] = customer_id
                st.session_state['current_page'] = 'Home'
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid email or password")

    st.button("New User?", on_click=lambda: st.session_state.update({'current_page': 'Register'}))
