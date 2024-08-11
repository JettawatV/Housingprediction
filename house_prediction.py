import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import plotly.express as px
import numpy as np

def load_model(zip_path, model_filename):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Extract the model file
        zip_ref.extractall()
    
    # Load the model from the extracted file
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    
    return model

# Load the model
model = load_model('second_model.zip', 'second_model.pkl')

# Load your datasets
dataset = pd.read_csv('housing.csv')  # Replace with your dataset path
average_increase = pd.read_csv('average_increase.csv')  # Replace with your average increase CSV path

# Streamlit app code
st.title('Housing Price Benchmark Value Prediction')

# Select Province
province = st.selectbox('Select Province', dataset['Province'].unique())

# Filter areas based on the selected province
areas_in_province = dataset[dataset['Province'] == province]['Area'].unique()
area = st.selectbox('Select Area', areas_in_province)

# Input fields
house_type = st.selectbox('Select House Type', dataset['House_Type'].unique())

# Slider for number of years
years = st.slider('Select Number of Years to Predict', min_value=1, max_value=10, value=1)

# Function to get average increase percentage based on user inputs
def get_average_increase(house_type, province, area):
    try:
        avg_increase_row = average_increase[
            (average_increase['House_Type'] == house_type) &
            (average_increase['Province'] == province) &
            (average_increase['Area'] == area)
        ]
        return avg_increase_row['Percentage_Increase'].values[0]
    except IndexError:
        st.error('No average increase data available for the selected inputs.')
        return None

average_increase_percentage = get_average_increase(house_type, province, area)

# Function to get the most recent HPI data
def get_latest_hpi(house_type, province, area):
    filtered_data = dataset[
        (dataset['House_Type'] == house_type) &
        (dataset['Province'] == province) &
        (dataset['Area'] == area)
    ]
    
    if filtered_data.empty:
        st.error('No data available for the selected inputs.')
        return None
    
    return filtered_data.iloc[-1]['HPI']

latest_hpi = get_latest_hpi(house_type, province, area)

# Default values for features other than HPI
default_values = {
    "Net temporary emigration": 0.0,
    "Net emigration": 0.0,
    "Emigrants": 0.79,
    "Shelter": 0.01,
    "Unemployment": 0.08,
    "Population": 0.04,
    "Labour force": 0.5,
    "Employment": 0.05,
    "Average income excluding zeros": 0.0
}

# Encode categorical data using OneHotEncoder
categorical_features = ['House_Type', 'Province', 'Area']
numerical_features = list(default_values.keys()) + ['HPI']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='constant', fill_value=0), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

if st.button('Predict Housing Price Value'):
    if latest_hpi is not None:
        if average_increase_percentage is not None:
            try:
                predictions = []
                years_range = list(range(2024, 2024 + years))
                
                for year in years_range:
                    # Adjust HPI for each year
                    adjusted_hpi = latest_hpi * (1 + average_increase_percentage / 100 * (year - 2024))
                    
                    # Create a DataFrame with the specified feature values
                    input_data_adjusted = pd.DataFrame({
                        'HPI': [adjusted_hpi],
                        **default_values,
                        'House_Type': [house_type],
                        'Province': [province],
                        'Area': [area]
                    })
                    
                    # Apply the preprocessor to the input data
                    input_data_preprocessed = preprocessor.fit_transform(input_data_adjusted)
                    
                    # Make prediction for the adjusted data
                    prediction = model.predict(input_data_preprocessed)[0]
                    predictions.append(prediction)
                
                # Plot predictions
                df_plot = pd.DataFrame({
                    'Year': years_range,
                    'Predicted Benchmark Value': predictions
                })
                
                fig = px.line(df_plot, x='Year', y='Predicted Benchmark Value', markers=True, title='Predicted Housing Price Benchmark Value Over Years')
                fig.update_traces(
                    mode='lines+markers+text',
                    text=[f'{value:.2f}' for value in df_plot['Predicted Benchmark Value']],
                    textposition='top center',
                    marker=dict(size=8, color='blue')
                )
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Predicted Housing Price Benchmark Value',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f'Error making prediction: {e}')
        else:
            st.error('Error: Average increase percentage is None.')
    else:
        st.error('Error: Latest HPI is None.')
