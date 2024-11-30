# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:13:12 2024

@author: kusha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Define the dataset
data = {
    "Size (sqft)": [850, 900, 1200, 1500, 1800, 2000, 850, 950, 1350, 1700],
    "Bedrooms": [2, 2, 3, 4, 3, 4, 1, 2, 3, 3],
    "Age (years)": [5, 10, 8, 15, 20, 7, 30, 25, 12, 18],
    "Price (USD)": [150000, 140000, 200000, 250000, 240000, 300000, 100000, 130000, 220000, 260000],
}

# Create DataFrame
df = pd.DataFrame(data)

# Define features and target
X = df[["Size (sqft)", "Bedrooms", "Age (years)"]]
y = df["Price (USD)"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KNN Regressor
k = 3
knn = KNeighborsRegressor(n_neighbors=k)

# Fit the model on training data
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Print results
print("Predicted Prices:", y_pred)
print("Actual Prices:", y_test.values)
print("Mean Squared Error:", mse)
