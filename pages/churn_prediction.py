import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.utils import register_keras_serializable
import tensorflow as tf

def show():


    st.title("Churn Prediction")

    # File uploader for the user to upload the dataset
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    
    if uploaded_file is not None:
        # Load the uploaded dataset
        input_data = pd.read_csv(uploaded_file)
        
        # Display the uploaded dataset
        st.write("Uploaded Dataset:")
        st.dataframe(input_data)
        @register_keras_serializable()
        def swish(x):
            return x * tf.keras.activations.sigmoid(x)
        # Load the trained model
        model_load_path = 'trained_churn_model.keras'  # Update this path
        loaded_model = load_model(model_load_path)

        # Process the dataset (Assuming 'Customer ID' needs to be kept and categorical columns need one-hot encoding)
        customer_ids = input_data['Customer ID']
        new_X = input_data.drop(['Customer ID'], axis=1)

        # One-hot encode categorical columns (modify accordingly based on your specific columns)
        categorical_cols = new_X.select_dtypes(include=['object']).columns
        new_X = pd.get_dummies(new_X, columns=categorical_cols, drop_first=True)

        # Ensure the columns align with the training dataset
        # Load the columns used for training from the training phase
        training_columns = [...]  # Replace with the actual list of training columns

        missing_cols = set(training_columns) - set(new_X.columns)
        for col in missing_cols:
            new_X[col] = 0
        new_X = new_X[training_columns]  # Reorder the columns

        # Normalize the input data using the same scaler (assumed to be saved as part of the training process)
        scaler = StandardScaler()
        new_X_scaled = scaler.fit_transform(new_X)

        # Predict churn probabilities
        predictions = loaded_model.predict(new_X_scaled)

        # Convert probabilities to binary labels (1 for churn, 0 for no churn)
        predicted_labels = (predictions > 0.5).astype(int)

        # Create a result DataFrame with Customer ID and Churn Label
        result_df = pd.DataFrame({
            'Customer ID': customer_ids,
            'Churn Label': predicted_labels.flatten()
        })

        # Display the prediction results
        st.write("Churn Prediction Results:")
        st.dataframe(result_df)

        # Option to download the results as a CSV file
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download prediction results as CSV",
            data=csv,
            file_name='churn_prediction_results.csv',
            mime='text/csv'
        )

        if st.button("Back to Admin Dashboard"):
      
       
            st.session_state['current_page'] = 'Admin'
            
            st.rerun()