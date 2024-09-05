import streamlit as st

def show():
    st.title("NextGen Telecom services")

    # Display the welcome message
    st.write(
        """
        Welcome to the Future of Telecom! 

        At NextGen Telecom services, we’re redefining connectivity with our next-generation telecom services designed just for you. Say goodbye to one-size-fits-all plans and hello to a personalized experience tailored to your unique needs. Whether you need high-speed data, unlimited calls, or exclusive features, we offer customized plans that grow with you.

        Explore a world where flexibility meets innovation, and enjoy seamless, reliable service that puts you in control. Discover the future of telecom today and experience connectivity like never before. 

        Join us and transform the way you stay connected!
        """
    )
    
    # Check if the user is logged in
    if st.session_state['is_admin']:
        if st.button("Go to Admin Page"):
            # Code to navigate to search page (if using Streamlit's routing, you can use st.experimental_rerun())
            st.session_state['current_page'] = 'Admin'  # Set the page state
            st.rerun()
    elif 'customer_id' in st.session_state:
        
        if st.button("Find your best plan"):
            # Code to navigate to search page (if using Streamlit's routing, you can use st.experimental_rerun())
            st.session_state['current_page'] = 'Search'  # Set the page state
            st.rerun()
    else:
        if st.button("Login"):
            # Code to navigate to login page (if using Streamlit's routing, you can use st.experimental_rerun())
            st.session_state['current_page'] = 'Login'  # Set the page state
            st.rerun()
