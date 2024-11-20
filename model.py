import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('phishing_urls.csv')

# Display the first few rows of the dataset to get an idea of its structure
print("First 5 rows of the dataset:")
print(df.head())

# Get more detailed information about the dataset: number of rows, columns, types of data, missing values
print("\nDataset Info:")
print(df.info())

# Check for any missing values in the dataset
print("\nMissing Values:")
print(df.isnull().sum())

# Get some basic statistics about numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Check for the distribution of the target variable (if present, e.g., 'is_phishing')
if 'is_phishing' in df.columns:
    print("\nTarget Variable Distribution:")
    print(df['is_phishing'].value_counts())

# Check for column names to ensure we know what features are present
print("\nColumn Names:")
print(df.columns)

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
# df = pd.read_csv('path_to_your_dataset.csv')  # Use this line if you have the dataset locally

# Preprocess the target label (status column)
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# Separate the features (X) and the target (y)
X = df.drop(columns=['url', 'status'])  # Dropping the URL and the target column
y = df['status']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data (Standard Scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print(X_train.shape) 

import joblib

# Save the model
joblib.dump(rf_model, 'phishing_url_model.pkl')
