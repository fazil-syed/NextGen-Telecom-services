import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.utils import register_keras_serializable
import tensorflow as tf

def show():


    st.title("Churn Prediction")

 
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    
    if uploaded_file is not None:

        input_data = pd.read_csv(uploaded_file)
        
 
        st.write("Uploaded Dataset:")
        st.dataframe(input_data)
        @register_keras_serializable()
        def swish(x):
            return x * tf.keras.activations.sigmoid(x)
   
        model_load_path = 'trained_churn_model.keras'
        loaded_model = load_model(model_load_path)

        customer_ids = input_data['Customer ID']
        new_X = input_data.drop(['Customer ID'], axis=1)

      
        categorical_cols = new_X.select_dtypes(include=['object']).columns
        new_X = pd.get_dummies(new_X, columns=categorical_cols, drop_first=True)

    
        training_columns = [...]  

        missing_cols = set(training_columns) - set(new_X.columns)
        for col in missing_cols:
            new_X[col] = 0
        new_X = new_X[training_columns]


        scaler = StandardScaler()
        new_X_scaled = scaler.fit_transform(new_X)

       
        predictions = loaded_model.predict(new_X_scaled)

        
        predicted_labels = (predictions > 0.5).astype(int)

        result_df = pd.DataFrame({
            'Customer ID': customer_ids,
            'Churn Label': predicted_labels.flatten()
        })


        st.write("Churn Prediction Results:")
        st.dataframe(result_df)

 
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