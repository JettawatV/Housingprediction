import streamlit as st
import pandas as pd
import joblib
import zipfile
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer,KNNImputer
import xgboost as xgb




# Function to load the compressed model
def load_zip_pipeline(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(file_name) as f:
            pipeline = pickle.load(f)
    return pipeline

# Load your dataset
dataset = pd.read_csv('housing.csv')  # Update with your dataset path

# Load the saved XGBoost model pipeline
zip_path = 'xgboost_pipeline.zip'  # Compressed file path
pipeline_file_name = 'xgboost_pipeline.pkl'
pipeline = load_zip_pipeline(zip_path, pipeline_file_name)

# Function to get features for prediction based on user input
# Function to get features for prediction based on user input
def get_features(house_type, province, area):
    # Filter dataset based on user input
    filtered_data = dataset[
        (dataset['House_Type'] == house_type) &
        (dataset['Province'] == province) &
        (dataset['Area'] == area)
    ]
    
    # Handle case where no data is found
    if filtered_data.empty:
        st.error('No data available for the selected inputs.')
        return None
    
    # Get the most recent data (or handle it accordingly)
    latest_data = filtered_data.iloc[-1]
    
    # Ensure all required columns are present
    required_columns = [
        'House_Type', 'Province', 'Area',
        'Benchmark', 'HPI', 'Aggregate income',
        'Average income excluding zeros', 'Median income excluding zeros',
        'Number with income', 'Prime rate'
    ]
    
    for column in required_columns:
        if column not in filtered_data.columns:
            st.error(f"Column '{column}' not found in the dataset.")
            return None
    
    # Create a DataFrame with the required features
    return pd.DataFrame({
        'House_Type': [house_type],
        'Province': [province],
        'Area': [area],
        'Benchmark': [latest_data['Benchmark']],
        'HPI': [latest_data['HPI']],
        'Aggregate income': [latest_data['Aggregate income']],
        'Average income excluding zeros': [latest_data['Average income excluding zeros']],
        'Median income excluding zeros': [latest_data['Median income excluding zeros']],
        'Number with income': [latest_data['Number with income']],
        'Prime rate': [latest_data['Prime rate']],
        # Add other features here as needed
    })

# Streamlit app layout
st.title('Benchmark Value Predictor')

# User input fields
house_type = st.selectbox('House Type', dataset['House_Type'].unique())
province = st.selectbox('Province', dataset['Province'].unique())
area = st.selectbox('Area', dataset['Area'].unique())
years = st.slider("Select Number of Years to Predict", 1, 10)

# Fetch features for prediction
input_data = get_features(house_type, province, area)

if input_data is not None:
    # Create a button for prediction
    if st.button('Predict Benchmark Value'):
        # Make prediction
        prediction = pipeline.predict(input_data)
        
        # Display prediction results
        st.write(f'Predicted Benchmark Value: ${prediction[0]:,.2f}')
