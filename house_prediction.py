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

# Load your dataset (update with your dataset path)
dataset = pd.read_csv('housing.csv')  # Replace with the correct path to your dataset

# Load the saved XGBoost model pipeline
zip_path = 'xgboost_pipeline.zip'  # Path to the compressed ZIP file
pipeline_file_name = 'xgboost_pipeline.pkl'  # Name of the file inside the ZIP
pipeline = load_zip_pipeline(zip_path, pipeline_file_name)

# Streamlit app code
st.title('Housing Benchmark Prediction')

# Input fields
house_type = st.selectbox('Select House Type', dataset['House_Type'].unique())
province = st.selectbox('Select Province', dataset['Province'].unique())
area = st.selectbox('Select Area', dataset['Area'].unique())

# Get features
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
    
    # Create a DataFrame with the required features
    return pd.DataFrame({
        'House_Type': [house_type],
        'Province': [province],
        'Area': [area],
        'Average income excluding zeros': [latest_data['Average income excluding zeros']],
        'Median income excluding zeros': [latest_data['Median income excluding zeros']],
        'Prime rate': [latest_data['Prime rate']],
        '5-year personal fixed term': [latest_data['5-year personal fixed term']],
        'Employment': [latest_data['Employment']],
        'Employment rate': [latest_data['Employment rate']],
        'Labour force': [latest_data['Labour force']],
        'Population': [latest_data['Population']],
        'Unemployment': [latest_data['Unemployment']],
        'Unemployment rate': [latest_data['Unemployment rate']],
        'All-items': [latest_data['All-items']],
        'Gasoline': [latest_data['Gasoline']],
        'Goods': [latest_data['Goods']],
        'Household operations, furnishings and equipment': [latest_data['Household operations, furnishings and equipment']],
        'Shelter': [latest_data['Shelter']],
        'Transportation': [latest_data['Transportation']],
        'Emigrants': [latest_data['Emigrants']],
        'Immigrants': [latest_data['Immigrants']],
        'Net emigration': [latest_data['Net emigration']],
        'Net non-permanent residents': [latest_data['Net non-permanent residents']],
        'Net temporary emigration': [latest_data['Net temporary emigration']],
        'Returning emigrants': [latest_data['Returning emigrants']],
        'Benchmark': [latest_data['Benchmark']],
        'HPI': [latest_data['HPI']]
    })

input_data = get_features(house_type, province, area)

# Predict button
if st.button('Predict Benchmark Value'):
    if input_data is not None:
        # Make prediction
        try:
            prediction = pipeline.predict(input_data)
            st.write(f'Predicted Benchmark Value: {prediction[0]}')
        except Exception as e:
            st.error(f'Error making prediction: {e}')
    else:
        st.error('Error: Input data is None.')
