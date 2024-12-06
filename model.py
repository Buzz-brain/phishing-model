import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('phishing_urls.csv')

# Display the first few rows of the dataset to verify it's loaded correctly
print(df.head())

# Display data types of all columns
print(df.dtypes)

# Check for missing values
print(df.isnull().sum())

# Drop the 'url' column since it's not useful
df = df.drop(columns=['url'])

# Map the target variable 'status' to numeric
df['status'] = df['status'].map({'legitimate': 0, 'phishing': 1})

# Display the first few rows again to confirm the changes
print(df.head())

# Features (all columns except 'status')
X = df.drop(columns=['status'])

# Target (status column)
y = df['status']

from sklearn.model_selection import train_test_split

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shape of the resulting sets to confirm the split
print(X_train.shape, X_test.shape)

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform both the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the scaled training data
rf_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report (precision, recall, f1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

import joblib
from sklearn.preprocessing import StandardScaler

# Save the trained model
joblib.dump(rf_model, 'phishing_model.pkl')

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

