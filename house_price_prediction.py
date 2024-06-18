import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and testing data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Check for missing values in training and testing data
print(train_data.isnull().sum())
print(test_data.isnull().sum())


# Identify columns with missing values and their data types
missing_cols_train = train_data.columns[train_data.isnull().any()]
missing_cols_test = test_data.columns[test_data.isnull().any()]

print("Missing columns in training data and their types:")
for col in missing_cols_train:
    print(f"{col}: {train_data[col].dtype}")

print("\nMissing columns in testing data and their types:")
for col in missing_cols_test:
    print(f"{col}: {test_data[col].dtype}")

# Handle missing values based on column data type
for col in missing_cols_train:
    if train_data[col].dtype == 'object':  # Handle string columns
        train_data[col] = train_data[col].fillna(train_data[col].mode()[0])  # Fill with most frequent value
    else:  # Handle numerical columns
        train_data[col] = train_data[col].fillna(train_data[col].mean())

for col in missing_cols_test:
    if test_data[col].dtype == 'object':  # Handle string columns
        test_data[col] = test_data[col].fillna(test_data[col].mode()[0])  # Fill with most frequent value
    else:  # Handle numerical columns
        test_data[col] = test_data[col].fillna(test_data[col].mean())

# Print column names to verify they match
print(train_data.columns)
print(test_data.columns)

# Define the features and the target variable for training data
train_features = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
train_target = train_data['SalePrice']

# Define the features for testing data (no target variable)
test_features = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]

# Convert to NumPy arrays (optional)
X_train = train_features.values
y_train = train_target.values
X_test = test_features.values

# Create the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict the prices for the test set
y_pred = model.predict(X_test)

# Save predictions to a CSV file
test_data['PredictedSalePrice'] = y_pred
test_data.to_csv('predicted_prices.csv', index=False)


# For demonstration purposes, we plot predicted prices vs index (without actual prices)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(y_pred)), y=y_pred)
plt.xlabel('Index')
plt.ylabel('Predicted Prices')
plt.title('Predicted Prices')
plt.show()
