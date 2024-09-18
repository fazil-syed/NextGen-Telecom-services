import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import euclidean
def load_data():
    df = pd.read_csv("dataset.csv")
    
   
    customer_ids = df['Customer ID']
    

    df.drop('Customer ID', axis=1, inplace=True)
    
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
 
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df, customer_ids

def quantify_preferences(df):
  
    df['Pref_Comm_Channel'] = df['Preferred Communication Channel'].map({'Email': 1, 'Phone': 2, 'SMS': 3})
    df['Pref_Comm_Time'] = df['Preferred Communication Time'].map({'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Anytime': 4})
    df['Pref_Services_Features'] = df['Preferred Services/Features'].map({'International Calling': 1, 'Data Rollover': 2, 'Family Plan': 3})
    df['Pref_Payment_Method'] = df['Preferred Payment Method'].map({'Bank Transfer': 1, 'Debit Card': 2, 'Online': 3, 'Credit Card': 4})
    
    df.drop(['Preferred Communication Channel', 'Preferred Communication Time', 'Preferred Services/Features', 'Preferred Payment Method'], axis=1, inplace=True)
    
    return df


def normalize_data(df):
    scaler = MinMaxScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    return df


def topsis_with_preferences(df, non_pref_criteria, weights):
   
    df_normalized = df[non_pref_criteria] / np.sqrt((df[non_pref_criteria]**2).sum())
    
 
    df_weighted = df_normalized * weights
    
    
    ideal_solution = df_weighted.max()
    negative_ideal_solution = df_weighted.min()
    
   
    df['distance_to_ideal'] = np.sqrt(((df_weighted - ideal_solution) ** 2).sum(axis=1))
    df['distance_to_negative_ideal'] = np.sqrt(((df_weighted - negative_ideal_solution) ** 2).sum(axis=1))
    
    
    df['relative_closeness'] = df['distance_to_negative_ideal'] / (df['distance_to_negative_ideal'] + df['distance_to_ideal'])
    
    df['rank'] = df['relative_closeness'].rank(ascending=False)
    
    return df


def decision_making_with_preferences(df, non_pref_criteria, weights):
    df_ranked = topsis_with_preferences(df, non_pref_criteria, weights)
    
  
    num_plans = 4
    df_ranked['BestService'] = pd.cut(df_ranked['rank'], bins=num_plans, labels=False, include_lowest=True)
    
    
    service_names = {0: 'Basic Plan', 1: 'Standard Plan', 2: 'Premium Plan', 3: 'Ultimate Plan'}
    df_ranked['BestServiceName'] = df_ranked['BestService'].map(service_names)
    
    return df_ranked


def predict_plan():
  
    df, customer_ids = load_data()
    non_pref_criteria = ['Data Usage (GB)', 'Voice Minutes Usage', 'SMS Usage']
    weights = [0.3, 0.3, 0.4]      
    df = quantify_preferences(df)
    df = normalize_data(df)
    
   
    df_ranked = decision_making_with_preferences(df, non_pref_criteria, weights)
    
   
    df_ranked['Customer ID'] = customer_ids
    
    
    df_ranked['Customer ID'] = df_ranked['Customer ID'].astype(str)
   
    df_ranked.to_csv('predicted_service_plans.csv', index=False)

    return df_ranked[['Customer ID', 'BestService', 'BestServiceName']]




def get_best_service_name_for_customer(customer_id):

    df_final = predict_plan()
    

    result = df_final[df_final['Customer ID'] == customer_id]
    
    if not result.empty:
        
        best_service_name = result['BestServiceName'].values[0]
        return best_service_name
    else:
        return f"Customer ID {customer_id} not found."
    

def add_new_customer(customerInput):
   
    df = pd.read_csv('dataset.csv')
 
    new_record = pd.DataFrame([customerInput])
    
   
    df = pd.concat([df, new_record], ignore_index=True)
    
  
    df.to_csv('dataset.csv', index=False)
    
   
    predict_plan()
    print(customerInput)
    print("Customer added successfully!")




def recommend_plan(age, gender, location, education_level):

    plans_df = pd.read_csv('plans_dataset.csv')
    

    input_data = pd.DataFrame([[age, gender, location, education_level]], 
                              columns=['Age', 'Gender', 'Location', 'Education Level'])
    
 
    label_encoders = {}
    for column in ['Gender', 'Location', 'Education Level']:
        le = LabelEncoder()
        plans_df[column] = le.fit_transform(plans_df[column])
        input_data[column] = le.transform(input_data[column])
        label_encoders[column] = le
    

    features = ['Age', 'Gender', 'Location', 'Education Level']
    similarity = cosine_similarity(input_data[features], plans_df[features])
    

    most_similar_index = similarity.argmax()
    

    recommended_plan = plans_df.iloc[most_similar_index]['BestServiceName']
    
    return recommended_plan
def convert_to_fuzzy(value, category):
    if category == 'age':
        if value <= 20:
            return (0.2, 0.7, 0.1)  
        elif 20 < value <= 35:
            return (0.7, 0.2, 0.1)  
        else:
            return (0.1, 0.8, 0.1)  
    elif category == 'gender':
        return (1.0, 0.0, 0.0) if value == 'Male' else (0.0, 1.0, 0.0)
    elif category == 'location':
        if value == 'Miami':
            return (0.8, 0.1, 0.1)
        elif value == 'New York':
            return (0.7, 0.2, 0.1)
        elif value == 'Chicago':
            return (0.6, 0.3, 0.1)
        elif value == 'Los Angeles':
            return (0.5, 0.4, 0.1)
        elif value == 'Houston':
            return (0.4, 0.5, 0.1)
        else:
            return (0.2, 0.6, 0.2)  
    elif category == 'education_level':
        if value == 'Bachelor':
            return (0.8, 0.1, 0.1)
        elif value == 'Master':
            return (0.6, 0.2, 0.2)
        else:
            return (0.3, 0.5, 0.2)


def flatten_fuzzy_values(fuzzy_values):
    return [item for sublist in fuzzy_values for item in sublist]

def recommend_plan_ifs(age, gender, location, education_level):
  
    user_prefs = [
        convert_to_fuzzy(age, 'age'),
        convert_to_fuzzy(gender, 'gender'),
        convert_to_fuzzy(location, 'location'),
        convert_to_fuzzy(education_level, 'education_level')
    ]
    

    user_prefs_flat = flatten_fuzzy_values(user_prefs)
    plans_df = pd.read_csv('plans_dataset.csv')

    plan_prefs = [
        flatten_fuzzy_values([
            convert_to_fuzzy(row['Age'], 'age'), convert_to_fuzzy(row['Gender'], 'gender'), 
            convert_to_fuzzy(row['Location'], 'location'), convert_to_fuzzy(row['Education Level'], 'education_level')
        ])
        for _, row in plans_df.iterrows()
    ]
    

    distances = [euclidean(user_prefs_flat, plan) for plan in plan_prefs]

    most_similar_index = np.argmin(distances)
    return plans_df.iloc[most_similar_index]['BestServiceName']




def recommend_plan_topsis(age, gender, location, education_level):
    scaler = MinMaxScaler()
    plans_df = pd.read_csv('plans_dataset.csv')

    input_data = pd.DataFrame([[age, gender, location, education_level]], 
                              columns=['Age', 'Gender', 'Location', 'Education Level'])
    
    label_encoders = {}
    for column in ['Gender', 'Location', 'Education Level']:
        le = LabelEncoder()
        plans_df[column] = le.fit_transform(plans_df[column])
        input_data[column] = le.transform(input_data[column])
        label_encoders[column] = le
    

    plan_prefs = plans_df[['Age', 'Gender', 'Location', 'Education Level']].values

    normalized_plans = scaler.fit_transform(plan_prefs)
    normalized_user = scaler.transform(input_data.to_numpy())  
   
    ideal_plan = np.max(normalized_plans, axis=0)
    negative_plan = np.min(normalized_plans, axis=0)
    
    distance_to_ideal = euclidean(normalized_user.flatten(), ideal_plan)
    distance_to_negative = euclidean(normalized_user.flatten(), negative_plan)
    

    scores = distance_to_negative / (distance_to_ideal + distance_to_negative)
    
    best_plan_index = np.argmax(scores)
    return plans_df.iloc[best_plan_index]['BestServiceName']


def predict_churn(customer_ids):
    churn_labels = np.random.randint(0, 2, size=len(customer_ids))

    return churn_labels 