# Step 1: Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Cache the data loading and model training
@st.cache_data
def load_and_preprocess_data(csv_path):
    """Loads data, performs one-hot encoding."""
    df = pd.read_csv(csv_path)
    df = pd.get_dummies(df, columns=['Extracurricular Activities'], drop_first=True) # Explicitly specify columns
    if 'Extracurricular Activities_Yes' in df.columns:
         df['Extracurricular Activities_Yes'] = df['Extracurricular Activities_Yes'].astype(int)
    return df

@st.cache_resource
def train_model(df):
    """Trains the Linear Regression model."""
    X = df.drop('Performance Index', axis=1)
    y = df['Performance Index']
    model = LinearRegression()
    model.fit(X, y) # Train on the full dataset for the app
    return model, X.columns

# --- Streamlit App ---
st.set_page_config(layout="wide") # Use wider layout
st.title('Student Performance Index Predictor')
st.image("Tracking_Headline.png", caption="Student Performance Tracking") # Add the image here

# Load data and train model
try:
    df_processed = load_and_preprocess_data('Student_Performance.csv')
    model, feature_names = train_model(df_processed.copy())

    # Create columns for layout
    col1, col2 = st.columns([0.6, 0.4]) # Adjust column width ratios if needed (e.g., 60% for inputs, 40% for output)

    with col1: # Input section
        st.header('Input Student Data:')
        # Define input widgets directly
        hours_studied = st.slider('Hours Studied', 1, 9, 5)
        previous_scores = st.slider('Previous Scores', 40, 99, 75)
        extracurricular = st.selectbox('Extracurricular Activities', ('No', 'Yes'))
        sleep_hours = st.slider('Sleep Hours', 4, 9, 7)
        sample_papers = st.slider('Sample Question Papers Practiced', 0, 9, 5)

    with col2: # Output section
        st.header('Prediction Result')
        # Add some vertical space for better alignment
        st.write("\n")
        st.write("\n")

        if st.button('Predict Performance Index'):
            try:
                # Gather input data inside the button click handler
                input_data = {
                    'Hours Studied': hours_studied,
                    'Previous Scores': previous_scores,
                    # Ensure the column name matches the one from get_dummies
                    'Extracurricular Activities_Yes': 1 if extracurricular == 'Yes' else 0,
                    'Sleep Hours': sleep_hours,
                    'Sample Question Papers Practiced': sample_papers
                }
                # Prepare input for prediction
                input_df = pd.DataFrame([input_data])
                # Ensure columns are in the same order as during training
                # Make sure feature_names includes 'Extracurricular Activities_Yes'
                # and not 'Extracurricular Activities'
                input_df = input_df[feature_names]

                prediction = model.predict(input_df)
                st.success(f'Predicted Performance Index: {prediction[0]:.2f}')
                st.balloons() # Add a little celebration
            except KeyError as ke:
                st.error(f"Feature mismatch error: {ke}. Check if all input features match the model's training features.")
            except Exception as pred_e:
                st.error(f"Error during prediction: {pred_e}")

except FileNotFoundError:
    st.error("Error: 'Student_Performance.csv' not found. Make sure the file is in the same directory.")
except Exception as e:
    st.error(f"An error occurred: {e}")
