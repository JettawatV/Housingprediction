import streamlit as st
import pandas as pd
import zipfile
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb

# Function to load the compressed model
def load_zip_pipeline(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(file_name) as f:
            model = pickle.load(f)
    return model

# Load your dataset (update with your dataset path)
dataset = pd.read_csv('housing.csv')  # Replace with the correct path to your dataset

# Load the saved XGBoost model and preprocessor
model = load_zip_pipeline('xgboost_pipeline.zip','xgboost_pipeline.pkl')

with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Define feature categories
numeric_features = [
    'Average income excluding zeros', 'Median income excluding zeros', 'Prime rate',
    '5-year personal fixed term', 'Employment', 'Employment rate', 'Labour force',
    'Population', 'Unemployment', 'Unemployment rate', 'All-items', 'Gasoline',
    'Goods', 'Household operations, furnishings and equipment', 'Shelter', 'Transportation',
    'Emigrants', 'Immigrants', 'Net emigration', 'Net non-permanent residents',
    'Net temporary emigration', 'Returning emigrants']

categorical_features = ['House_Type', 'Area', 'Province']

# Streamlit app code
st.title('Housing Benchmark Prediction')

# Input fields
house_type = st.selectbox('Select House Type', dataset['House_Type'].unique())
province = st.selectbox('Select Province', dataset['Province'].unique())
area = st.selectbox('Select Area', dataset['Area'].unique())

# Function to get features based on user input
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
    })

# Get input data based on user selection
input_data = get_features(house_type, province, area)

# Predict button
if st.button('Predict Benchmark Value'):
    if input_data is not None:
        try:
            # Transform input data using the preprocessor
            input_data_processed = preprocessor.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_data_processed)
            st.write(f'Predicted Benchmark Value: {prediction[0]}')
        except Exception as e:
            st.error(f'Error making prediction: {e}')
    else:
        st.error('Error: Input data is None.')
