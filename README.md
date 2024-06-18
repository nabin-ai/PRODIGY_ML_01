# PRODIGY_ML_01
# House Price Prediction Model

This project implements a linear regression model to predict house prices based on their square footage (GrLivArea), number of bedrooms, and number of bathrooms. The model is trained using a training dataset (`train.csv`) and predictions are made for the testing dataset (`test.csv`).

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [License](#license)

## Project Description
The goal of this project is to build a linear regression model that can predict house prices using features such as square footage (GrLivArea), number of bedrooms (BedroomAbvGr), and number of bathrooms (FullBath and HalfBath). The model is trained on the training dataset and used to predict prices for the testing dataset.

## Dataset
The datasets used in this project are:
- `train.csv`: The training dataset containing features and target variable.
- `test.csv`: The testing dataset used to predict house prices.

Both files should be in CSV format and include the following columns:
- `GrLivArea`: The above ground living area square footage.
- `BedroomAbvGr`: The number of bedrooms above grade.
- `FullBath`: The number of full bathrooms.
- `HalfBath`: The number of half bathrooms.

Additionally, `train.csv` should include:
- `SalePrice`: The price of the house (target variable).

## Requirements
- Python 3.6 or higher
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/nabin-ai/house-price-prediction.git
    cd house-price-prediction
    ```

2. Install the required packages:
    ```sh
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3. Place the `train.csv` and `test.csv` files in the project directory.

## Usage
Run the following script to train the model and predict house prices for the test set:
```python
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

# Drop or fill missing values if necessary
train_data = train_data.dropna()
test_data = test_data.dropna()

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

# If you have the actual prices for the test set, load them for evaluation
# actual_prices = pd.read_csv('actual_prices.csv')  # Make sure this file exists and contains actual prices
# y_test = actual_prices['SalePrice'].values

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# print(f'Root Mean Squared Error: {rmse}')

# Plotting actual vs predicted prices (if actual prices are available)
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=y_pred)
# plt.xlabel('Actual Prices')
# plt.ylabel('Predicted Prices')
# plt.title('Actual Prices vs Predicted Prices')
# plt.show()

# For demonstration purposes, we plot predicted prices vs index (without actual prices)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=range(len(y_pred)), y=y_pred)
plt.xlabel('Index')
plt.ylabel('Predicted Prices')
plt.title('Predicted Prices')
plt.show()
