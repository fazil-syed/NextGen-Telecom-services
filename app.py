import streamlit as st

# Navigation function
def navigate(page):
    st.session_state['current_page'] = page

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False

# Sidebar for navigation
if st.session_state['logged_in']:
    st.sidebar.button("Home", on_click=navigate, args=("Home",))
    if not st.session_state['is_admin']:
        st.sidebar.button("Search", on_click=navigate, args=("Search",))
    st.sidebar.button("Logout", on_click=navigate, args=("Login",))

# Load the appropriate page
if st.session_state['current_page'] == 'Login':
    import pages.login as login
    login.show()
elif st.session_state['current_page'] == 'Register':
    import pages.register as register
    register.show()
elif st.session_state['current_page'] == 'Home':
    import pages.home as home
    home.show()
elif st.session_state['current_page'] == 'Search':
    import pages.search as search
    search.show()
elif st.session_state['current_page'] == 'Admin':
    import pages.admin as admin
    admin.show()
