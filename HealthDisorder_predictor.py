import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the Data
# For this example, we'll create a synthetic dataset
# Replace this part with your actual dataset loading code
data = {
    'Symptom1': [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
    'Symptom2': [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
    'Symptom3': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    'Disease': ['Disease1', 'Disease2', 'Disease1', 'Disease2', 'Disease1', 'Disease1', 'Disease2', 'Disease2', 'Disease1', 'Disease2']
}

df = pd.DataFrame(data)

# Step 2: Preprocess the Data
# Encode the target variable
df['Disease'] = df['Disease'].astype('category').cat.codes

# Split features and target
X = df.drop('Disease', axis=1)
y = df['Disease']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Step 5: Make Predictions
# Example of predicting a new sample
new_sample = np.array([[1, 0, 1]])  # Replace with actual symptoms
new_sample_scaled = scaler.transform(new_sample)
prediction = model.predict(new_sample_scaled)
predicted_disease = pd.Categorical(df['Disease']).categories[prediction][0]
print(f'Predicted Disease: {predicted_disease}')
