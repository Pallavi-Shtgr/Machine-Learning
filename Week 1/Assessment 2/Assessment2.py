
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load the dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Step 3: Data Cleaning
# Drop irrelevant columns for prediction
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Handle missing values: fill missing 'Age' with median, 'Embarked' with mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Step 4: String Manipulation
# Convert string columns to lowercase
df['Sex'] = df['Sex'].str.lower().str.strip()
df['Embarked'] = df['Embarked'].str.lower().str.strip()

# Convert categorical variables (like 'Sex' and 'Embarked') to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 5: Use NumPy for basic statistics
# Convert 'Age' and 'Fare' columns to NumPy arrays
ages = np.array(df['Age'])
fares = np.array(df['Fare'])

# Calculate mean and median for Age and Fare
mean_age = np.mean(ages)
median_age = np.median(ages)
mean_fare = np.mean(fares)
median_fare = np.median(fares)

print(f"Mean Age: {mean_age}, Median Age: {median_age}")
print(f"Mean Fare: {mean_fare}, Median Fare: {median_fare}")

# Step 6: Data Splitting
# Features (X) and Target (y)
X = df.drop('Survived', axis=1)  # Features (all columns except 'Survived')
y = df['Survived']  # Target variable (Survived)

# Split the dataset into 80% training and 20% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Build a Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Optional: Additional evaluation metrics (Confusion Matrix, Classification Report)
from sklearn.metrics import confusion_matrix, classification_report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
