
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Create a sample dataset (You can replace this with real data)
np.random.seed(42)
data_size = 1000

# Create sample data with features 'age', 'tenure', and target 'churn'
data = {
    'age': np.random.randint(18, 70, data_size),  
    'tenure': np.random.randint(1, 10, data_size), 
    'churn': np.random.randint(0, 2, data_size)  # Churn: 0 (no) or 1 (yes)
}

# Step 2: Create a DataFrame
df = pd.DataFrame(data)

# Step 3: Data Aggregation
# Group by 'churn' and aggregate to find the average age and tenure for churned and non-churned customers
churn_aggregated = df.groupby('churn').agg({
    'age': 'mean',
    'tenure': 'mean'
}).reset_index()

print("Aggregated Data (Average Age & Tenure of Customers based on Churn):")
print(churn_aggregated)

# Step 4: Data Splitting
# Features (X) and Target (y)
X = df[['age', 'tenure']]  # Features
y = df['churn']  # Target variable (churn)

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model Training
# Initialize and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Model Evaluation
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Additional Evaluation: Confusion Matrix and Classification Report
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Classification report provides precision, recall, f1-score for detailed performance analysis
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)
