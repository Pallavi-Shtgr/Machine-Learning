 		# ASSIGNMET 3 - Predicting House Prices 🏠💰

# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Loading the Dataset
boston = load_boston()  # This will raise a warning; consider using fetch_california_housing instead
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Exploring the Data
print(data.head())

# Visualizing Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Scatter Plot of RM vs PRICE
plt.figure(figsize=(6, 4))
plt.scatter(data['RM'], data['PRICE'])
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("House Price")
plt.title("RM vs Price")
plt.show()

# Checking for Missing Values
print(data.isnull().sum())

# Data Preparation
X = data.drop('PRICE', axis=1)  # Features
y = data['PRICE']  # Target variable

# Standardizing the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing the Models
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)

# Training the Models
lr_model.fit(X_train, y_train)
dt_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Making Predictions
lr_pred = lr_model.predict(X_test)
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluating the Models
models = {
    "Linear Regression": lr_pred,
    "Decision Tree": dt_pred,
    "Random Forest": rf_pred
}

for name, pred in models.items():
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")  
  
