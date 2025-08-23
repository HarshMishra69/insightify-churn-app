# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

print("--- Starting Model Training ---")

# --- Load the Dataset ---
try:
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("\n[ERROR] 'WA_Fn-UseC_-Telco-Customer-Churn.csv' not found.")
    print("Please download the dataset from Kaggle and place it in the 'insightify_project' folder.")
    exit()

# --- Data Cleaning and Preparation ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print("Data cleaning complete.")

# --- Define Features (X) and Target (y) ---
X = df.drop('Churn', axis=1)
y = df['Churn']

# --- Preprocessing Steps ---
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- Create and Train the Model using a Pipeline ---
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training the model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete!")

# --- Save (Pickle) the Trained Model ---
with open('churn_model.pkl', 'wb') as file:
    pickle.dump(model_pipeline, file)

print("\n--- ✅ Model Saved Successfully! ---")
print("The trained model is now saved as 'churn_model.pkl'.")
