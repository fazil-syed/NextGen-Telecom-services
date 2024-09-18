# Required libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
import xgboost as xgb

# Load the dataset
dataset_path = 'dataset.csv'
dataset = pd.read_csv(dataset_path)

# Preprocessing: Splitting the target and features
y = dataset['Churn Label'].values  # Target variable
X = dataset.drop(['Churn Label', 'Customer ID'], axis=1)  # Drop target and irrelevant columns

# Handling categorical variables with OneHotEncoding
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Scaling the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------------------
# PART 1: Support Vector Machine (SVM) Implementation
# ----------------------------------------------------------------------------------------

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predict on the test data using SVM
y_pred_svm = svm_classifier.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print("Support Vector Machine (SVM) Results:")
print(f"Accuracy: {accuracy_svm * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# ----------------------------------------------------------------------------------------
# PART 2: XGBoost Implementation
# ----------------------------------------------------------------------------------------

# Initialize and train the XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_classifier.fit(X_train, y_train)

# Predict on the test data using XGBoost
y_pred_xgb = xgb_classifier.predict(X_test)

# Evaluate the XGBoost model
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("\nXGBoost Results:")
print(f"Accuracy: {accuracy_xgb * 100:.2f}%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
print("Classification Report:")
print(classification_report(y_test, y_pred_xgb))
