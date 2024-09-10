import streamlit as st


st.set_page_config(
    layout="centered",  
    initial_sidebar_state="collapsed",  
)


def navigate(page):
    st.session_state['current_page'] = page

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Home'

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if 'is_admin' not in st.session_state:
    st.session_state['is_admin'] = False


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
elif st.session_state['current_page'] == 'PlanAndAnalytics':
    import pages.plan_details_analytics as plan_details_analytics
    plan_details_analytics.show()

elif st.session_state['current_page'] == 'ChurnPrediction':
    import pages.churn_prediction as churn_prediction
    churn_prediction.show()
