import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the dataset
def load_data():
    df = pd.read_csv("dataset.csv")
    
    # Keep the Customer ID column to merge later
    customer_ids = df['Customer ID']
    
    # Drop other columns not needed for preference analysis
    df.drop('Customer ID', axis=1, inplace=True)
    
    # Handle missing values separately for numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Fill missing values for numeric columns with mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill missing values for categorical columns with mode
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df, customer_ids

def quantify_preferences(df):
    # Convert categorical columns to numerical values
    df['Pref_Comm_Channel'] = df['Preferred Communication Channel'].map({'Email': 1, 'Phone': 2, 'SMS': 3})
    df['Pref_Comm_Time'] = df['Preferred Communication Time'].map({'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Anytime': 4})
    df['Pref_Services_Features'] = df['Preferred Services/Features'].map({'International Calling': 1, 'Data Rollover': 2, 'Family Plan': 3})
    df['Pref_Payment_Method'] = df['Preferred Payment Method'].map({'Bank Transfer': 1, 'Debit Card': 2, 'Online': 3, 'Credit Card': 4})
    
    # Drop original preference columns
    df.drop(['Preferred Communication Channel', 'Preferred Communication Time', 'Preferred Services/Features', 'Preferred Payment Method'], axis=1, inplace=True)
    
    return df

# Normalize the data
def normalize_data(df):
    scaler = MinMaxScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df

# Define the TOPSIS method with customer preferences and non-preferences
def topsis_with_preferences(df, non_pref_criteria, weights):
    # Normalize the decision matrix
    df_normalized = df[non_pref_criteria] / np.sqrt((df[non_pref_criteria]**2).sum())
    
    # Apply weights
    df_weighted = df_normalized * weights
    
    # Determine ideal and negative-ideal solutions
    ideal_solution = df_weighted.max()
    negative_ideal_solution = df_weighted.min()
    
    # Calculate separation measures
    df['distance_to_ideal'] = np.sqrt(((df_weighted - ideal_solution) ** 2).sum(axis=1))
    df['distance_to_negative_ideal'] = np.sqrt(((df_weighted - negative_ideal_solution) ** 2).sum(axis=1))
    
    # Calculate the relative closeness to the ideal solution
    df['relative_closeness'] = df['distance_to_negative_ideal'] / (df['distance_to_negative_ideal'] + df['distance_to_ideal'])
    
    # Rank the service plans
    df['rank'] = df['relative_closeness'].rank(ascending=False)
    
    return df

# Implement the decision-making process with preferences and non-preferences
def decision_making_with_preferences(df, non_pref_criteria, weights):
    df_ranked = topsis_with_preferences(df, non_pref_criteria, weights)
    
    # Map ranks to the service plans (0 to 3)
    num_plans = 4
    df_ranked['BestService'] = pd.cut(df_ranked['rank'], bins=num_plans, labels=False, include_lowest=True)
    
    # Map service indices to service names
    service_names = {0: 'Basic Plan', 1: 'Standard Plan', 2: 'Premium Plan', 3: 'Ultimate Plan'}
    df_ranked['BestServiceName'] = df_ranked['BestService'].map(service_names)
    
    return df_ranked

# Predict the best service plan and save the results
def predict_plan():
    # Load and preprocess dataset
    df, customer_ids = load_data()
    non_pref_criteria = ['Data Usage (GB)', 'Voice Minutes Usage', 'SMS Usage']
    weights = [0.3, 0.3, 0.4]  #  weights for non-preference criteria
    
    df = quantify_preferences(df)
    df = normalize_data(df)
    
    # Apply decision-making model
    df_ranked = decision_making_with_preferences(df, non_pref_criteria, weights)
    
    # Add Customer ID back to the dataframe
    df_ranked['Customer ID'] = customer_ids
    
    # Convert Customer ID to string
    df_ranked['Customer ID'] = df_ranked['Customer ID'].astype(str)
    # Save the results to a new CSV file
    df_ranked.to_csv('predicted_service_plans.csv', index=False)

    return df_ranked[['Customer ID', 'BestService', 'BestServiceName']]




def get_best_service_name_for_customer(customer_id):
    # Execute the prediction process
    df_final = predict_plan()
    
    # Find the specific Customer ID
    result = df_final[df_final['Customer ID'] == customer_id]
    
    if not result.empty:
        # Return the BestServiceName for that Customer ID
        best_service_name = result['BestServiceName'].values[0]
        return best_service_name
    else:
        return f"Customer ID {customer_id} not found."
    

def add_new_customer(customerInput):
    # Load the existing dataset
    df = pd.read_csv('dataset.csv')
    
    # Convert customerInput dictionary to DataFrame
    new_record = pd.DataFrame([customerInput])
    
    # Append the new record to the dataframe
    df = pd.concat([df, new_record], ignore_index=True)
    
    # Save the updated dataset back to the file
    df.to_csv('dataset.csv', index=False)
    
    # Call the predict_plan function on the updated dataset
    predict_plan()
    print(customerInput)
    print("Customer added successfully!")




def recommend_plan(age, gender, location, education_level):
    # Load the plans dataset
    plans_df = pd.read_csv('plans_dataset.csv')
    
    # Prepare the input features of the new user
    input_data = pd.DataFrame([[age, gender, location, education_level]], 
                              columns=['Age', 'Gender', 'Location', 'Education Level'])
    
    # Encode the categorical variables in the dataset for similarity matching
    label_encoders = {}
    for column in ['Gender', 'Location', 'Education Level']:
        le = LabelEncoder()
        plans_df[column] = le.fit_transform(plans_df[column])
        input_data[column] = le.transform(input_data[column])
        label_encoders[column] = le
    
    # Calculate similarity between the new user and the customers in the dataset
    features = ['Age', 'Gender', 'Location', 'Education Level']
    similarity = cosine_similarity(input_data[features], plans_df[features])
    
    # Find the index of the most similar customer
    most_similar_index = similarity.argmax()
    
    # Recommend the plan of the most similar customer
    recommended_plan = plans_df.iloc[most_similar_index]['BestServiceName']
    
    return recommended_plan
