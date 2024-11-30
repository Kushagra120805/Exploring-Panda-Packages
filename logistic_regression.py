# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:15:14 2024

@author: kusha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Define the dataset
data = {
    "Size (sqft)": [850, 900, 1200, 1500, 1800, 2000, 850, 950, 1350, 1700],
    "Bedrooms": [2, 2, 3, 4, 3, 4, 1, 2, 3, 3],
    "Age (years)": [5, 10, 8, 15, 20, 7, 30, 25, 12, 18],
    "Price (USD)": [150000, 140000, 200000, 250000, 240000, 300000, 100000, 130000, 220000, 260000]
}

# Create DataFrame
df = pd.DataFrame(data)

# Create a binary target column for price category
df['Price_Category'] = (df['Price (USD)'] > 200000).astype(int)

# Define features and target
X = df[["Size (sqft)", "Bedrooms", "Age (years)"]]
y = df["Price_Category"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and fit the Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = log_reg.predict(X_test_scaled)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Accuracy: {accuracy:.2f}")
