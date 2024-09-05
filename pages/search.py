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
        df_predictions = pd.read_csv('predicted_service_plans.csv')
        df_dataset = pd.read_csv('dataset.csv')

        # Filter the DataFrames to get the row with the matching CustomerID
        result_prediction = df_predictions[df_predictions['Customer ID'] == customerid]
        result_dataset = df_dataset[df_dataset['Customer ID'] == customerid]

        if not result_prediction.empty and not result_dataset.empty:
            # Display the basic details
            st.write("**Basic Details:**")
            row_prediction = result_prediction.iloc[0]
            st.write(f"**Best Service Recommended:** {row_prediction['BestServiceName']}")
            st.write(f"**Plan Type:** {result_dataset.iloc[0]['Plan Type']}")
            st.write(f"**Subscription Duration (months):** {result_dataset.iloc[0]['Subscription Duration (months)']}")
            st.write(f"**Monthly Charges:** {result_dataset.iloc[0]['Monthly Charges']}")
            st.write(f"**Data Usage (GB):** {result_dataset.iloc[0]['Data Usage (GB)']}")
            st.write(f"**Voice Minutes Usage:** {result_dataset.iloc[0]['Voice Minutes Usage']}")
            st.write(f"**SMS Usage:** {result_dataset.iloc[0]['SMS Usage']}")

            # Toggle button for additional details
            if 'show_more' not in st.session_state:
                st.session_state.show_more = False

            if st.button("View More Details"):
                st.session_state.show_more = not st.session_state.show_more

            if st.session_state.show_more:
                st.write("**Additional Details:**")
                # Display additional details
                for column, value in result_dataset.iloc[0].items():
                    if column not in ['Customer ID', 'Plan Type', 'Subscription Duration (months)', 'Monthly Charges', 'Data Usage (GB)', 'Voice Minutes Usage', 'SMS Usage']:
                        st.write(f"**{column}:** {value}")
        else:
            st.write("No customer found with the provided ID.")
    else:
        st.error("No customer ID found in session state.")
