import streamlit as st

def show():
    st.title("NextGen Telecom services")

   
    st.write(
        """
        Welcome to the Future of Telecom! 

        At NextGen Telecom services, we’re redefining connectivity with our next-generation telecom services designed just for you. Say goodbye to one-size-fits-all plans and hello to a personalized experience tailored to your unique needs. Whether you need high-speed data, unlimited calls, or exclusive features, we offer customized plans that grow with you.

        Explore a world where flexibility meets innovation, and enjoy seamless, reliable service that puts you in control. Discover the future of telecom today and experience connectivity like never before. 

        Join us and transform the way you stay connected!
        """
    )
    
    
    if st.session_state['is_admin']:
        if st.button("Go to Admin Page"):
          
            st.session_state['current_page'] = 'Admin'  
            st.rerun()
    elif 'customer_id' in st.session_state:
        
        if st.button("Find your best plan"):
            
            st.session_state['current_page'] = 'Search'  
            st.rerun()
    else:
        if st.button("Login"):
            
            st.session_state['current_page'] = 'Login'  
            st.rerun()
