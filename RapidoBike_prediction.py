import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Load Data
# Replace 'your_dataset.csv' with your actual dataset file
data = pd.read_csv('your_dataset.csv', parse_dates=['timestamp'])

# Data Preprocessing and Feature Engineering
def preprocess_data(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    
    # Drop the original timestamp column if not needed anymore
    df.drop(columns=['timestamp'], inplace=True)
    
    return df

data = preprocess_data(data)

# Define Features and Target
X = data.drop(columns=['ride_requests'])
y = data['ride_requests']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')

# Prediction for a Particular Hour
# Define a function to make predictions for a given timestamp
def predict_ride_requests(model, timestamp):
    ts = pd.to_datetime(timestamp)
    features = {
        'hour': ts.hour,
        'day': ts.day,
        'month': ts.month,
        'year': ts.year,
        'dayofweek': ts.dayofweek
    }
    features_df = pd.DataFrame([features])
    return model.predict(features_df)

# Example prediction for a specific hour
timestamp = '2024-05-22 14:00:00'  # Change to your desired timestamp
predicted_requests = predict_ride_requests(model, timestamp)
print(f'Predicted ride requests for {timestamp}: {predicted_requests[0]}')

# Plotting Actual vs Predicted for Test Set
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.xlabel('Test Set Index')
plt.ylabel('Ride Requests')
plt.title('Actual vs Predicted Ride Requests')
plt.show()
