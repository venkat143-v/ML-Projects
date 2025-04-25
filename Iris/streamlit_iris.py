import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load your dataset
data_path = "IRIS.csv"  # Replace with your dataset path
data = pd.read_csv(data_path)

# Split the data into features and target
X = data.iloc[:, :-1]  # Assuming all columns except the last are features
y = data.iloc[:, -1]   # Assuming the last column is the target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a K-Nearest Neighbors Classifier
model = KNeighborsClassifier(n_neighbors=4)
model.fit(X_train, y_train)

# Streamlit UI
st.title("Iris Dataset Classifier")
st.image("C:/AI/ML/9a32d741-51c7-4573-9799-8d933ee642c6.png", caption="Iris Flower Types") # Add the image here

st.sidebar.header("Model Configuration")
n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 10, 4)

# Update the model with the selected number of neighbors
model = KNeighborsClassifier(n_neighbors=n_neighbors)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Input for prediction
st.header("Make a Prediction")
sepal_length = st.number_input("Sepal Length", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, step=0.1)

if st.button("Predict"):
    try:
        sample = [[sepal_length, sepal_width, petal_length, petal_width]]
        sample_df = pd.DataFrame(sample, columns=X_train.columns)
        predicted_class = model.predict(sample_df)
        st.write(f"Predicted Class: {predicted_class[0]}")
    except Exception as e:
        st.error(f"Error in input: {e}")
