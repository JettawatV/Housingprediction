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
        'Average income excluding zeros', 'Median income excluding zeros',
        'Prime rate', '5-year personal fixed term', 'Employment',
        'Employment rate', 'Labour force', 'Population', 'Unemployment',
        'Unemployment rate', 'All-items', 'Gasoline', 'Goods',
        'Household operations, furnishings and equipment', 'Shelter',
        'Transportation', 'Emigrants', 'Immigrants', 'Net emigration',
        'Net non-permanent residents', 'Net temporary emigration',
        'Returning emigrants', 'Benchmark', 'HPI'
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
