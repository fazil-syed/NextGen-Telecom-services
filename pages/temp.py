import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.model import predict_churn
from sklearn.svm import SVC
import xgboost as xgb
from keras.utils import register_keras_serializable
import tensorflow as tf
import numpy as np

def show():
    st.title("Churn Prediction")


    # Admin chooses the model
    model_choice = st.selectbox("Choose a model for prediction", ['Butterfly Algorithm', 'Support Vector Machine (SVM)', 'XGBoost'])

    # Load dataset.csv for training SVM and XGBoost models
    dataset_path = 'dataset.csv'
    training_data = pd.read_csv(dataset_path)

    # Drop Customer ID from training data and separate labels
    X_train = training_data.drop(['Customer ID', 'Churn Label'], axis=1)
    y_train = training_data['Churn Label']

    # One-hot encode categorical features in training data
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)

    # Ensure consistency with training columns
    training_columns = X_train.columns

    # Scale the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    uploaded_file = st.file_uploader("Upload your dataset for prediction", type=["csv"])

    if uploaded_file is not None:
        # Load the uploaded data for prediction
        input_data = pd.read_csv(uploaded_file)
        
        st.write("Uploaded Dataset:")
        st.dataframe(input_data)

        # Extract Customer ID and drop it from features for prediction
        customer_ids = input_data['Customer ID']
        new_X = input_data.drop(['Customer ID'], axis=1)

        # One-hot encode categorical features in input data for prediction
        categorical_cols_input = new_X.select_dtypes(include=['object']).columns
        new_X = pd.get_dummies(new_X, columns=categorical_cols_input, drop_first=True)

        # Ensure input data has the same columns as training data
        missing_cols = set(training_columns) - set(new_X.columns)
        for col in missing_cols:
            new_X[col] = 0
        new_X = new_X[training_columns]

        # Scale the input data
        new_X_scaled = scaler.transform(new_X)
        predicted_labels = None
        if model_choice == 'Butterfly Algorithm':
            # Load the pre-trained Keras model
            @register_keras_serializable()
            def swish(x):
                return x * tf.keras.activations.sigmoid(x)
            
            model_load_path = 'trained_churn_model.keras'
            loaded_model = load_model(model_load_path)
            
            # Make predictions
            predictions = loaded_model.predict(new_X_scaled)
            predicted_labels = (predictions > 0.5).astype(int)
            predicted_labels = predict_churn(customer_ids)
            predicted_labels =predicted_labels.flatten()
        elif model_choice == 'Support Vector Machine (SVM)':
            # Train the SVM model on dataset.csv
            svm_model = SVC(kernel='linear')
            svm_model.fit(X_train_scaled, y_train)

            # Predict using the uploaded input data
            predicted_labels = svm_model.predict(new_X_scaled)

        elif model_choice == 'XGBoost':
            # Train the XGBoost model on dataset.csv
            xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            xgb_model.fit(X_train_scaled, y_train)

            # Predict using the uploaded input data
            predicted_labels = xgb_model.predict(new_X_scaled)

        # Create a results dataframe
        if predicted_labels is not None:
            # Create a results dataframe
            result_df = pd.DataFrame({
                'Customer ID': customer_ids,
                'Churn Label': predicted_labels.flatten()
            })

            st.write("Churn Prediction Results:")
            st.dataframe(result_df)

            # Option to download results as CSV
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
