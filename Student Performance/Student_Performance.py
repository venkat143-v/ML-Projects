# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
df = pd.read_csv('Student_Performance.csv')
print("--- Dataset Head ---")
print(df.head())

# Step 3: Explore the Data (Optional: Can be uncommented if needed)
# print("--- Dataset Info ---")
# print(df.info())
# print("--- Dataset Description ---")
# print(df.describe())
# print("--- Pairplot --- ")
# sns.pairplot(df)
# plt.show()

# Step 3a: Handle Categorical Features
# Explicitly handle 'Extracurricular Activities'
df = pd.get_dummies(df, columns=['Extracurricular Activities'], drop_first=True)
# Ensure the new boolean column is integer type
if 'Extracurricular Activities_Yes' in df.columns:
    df['Extracurricular Activities_Yes'] = df['Extracurricular Activities_Yes'].astype(int)

print("--- Data after One-Hot Encoding ---")
print(df.head())

# Step 4: Prepare Features & Target
X = df.drop('Performance Index', axis=1)  # Features
y = df['Performance Index']               # Target

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make Predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the Model
print("--- Model Evaluation ---")
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Step 9: Visualize Results
print("--- Visualizing Results --- ")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Diagonal line
plt.xlabel('Actual Performance Index')
plt.ylabel('Predicted Performance Index')
plt.title('Actual vs Predicted Performance Index')
plt.grid(True)
plt.show()
