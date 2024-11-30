import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"C:\Users\kusha\OneDrive\Desktop\house_prices.csv")

# Define features and target
X = data[['Size (sqft)', 'Bedrooms', 'Age (years)']]
y = data['Price (USD)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and calculate overall R² score
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print(f"Overall R² Score: {accuracy:.2f}")

# Calculate R² score for each feature individually
for feature in X.columns:
    X_single = data[[feature]]
    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single, y, test_size=0.2, random_state=42
    )
    model.fit(X_train_single, y_train_single)
    y_pred_single = model.predict(X_test_single)
    feature_accuracy = r2_score(y_test_single, y_pred_single)
    print(f"R² Score for {feature}: {feature_accuracy:.2f}")

# Plot each feature against the target
for feature in X.columns:
    plt.figure(figsize=(6, 4))
    plt.scatter(data[feature], data['Price (USD)'], color='blue', alpha=0.6)
    plt.title(f'{feature} vs Price (USD)')
    plt.xlabel(feature)
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()
