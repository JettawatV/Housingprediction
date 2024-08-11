import streamlit as st
import pandas as pd
import zipfile
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

# Function to load the compressed model
def load_zip_pipeline(zip_path, file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(file_name) as f:
            model = pickle.load(f)
    return model

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

model = load_zip_pipeline('xgboost_pipeline.zip', 'xgboost_pipeline.pkl')

# Load your datasets
dataset = pd.read_csv('housing.csv')  # Replace with your dataset path
average_increase = pd.read_csv('average_increase.csv')  # Replace with your average increase CSV path

# Streamlit app code
st.title('Housing Benchmark Prediction')

# Input fields
house_type = st.selectbox('Select House Type', dataset['House_Type'].unique())
province = st.selectbox('Select Province', dataset['Province'].unique())
area = st.selectbox('Select Area', dataset['Area'].unique())

# Slider for number of years
years = st.slider('Select Number of Years to Predict', min_value=1, max_value=10, value=1)

# Get average increase for the given house type
def get_average_increase(house_type, province, area):
    try:
        return average_increase[
            (average_increase['House_Type'] == house_type) &
            (average_increase['Province'] == province) &
            (average_increase['Area'] == area)
        ]['Percentage_Increase'].values[0]
    except IndexError:
        st.error('No average increase data available for the selected inputs.')
        return None

average_increase_percentage = get_average_increase(house_type, province, area)

# Growth rates for each feature
growth_rates = {
    'Prime rate': 0.005,  # 0.5% per year
    '5-year personal fixed term': 0.007,  # 0.7% per year
    'Employment': 0.01,  # 1% per year
    'Population': 0.01,  # 1% per year
    'Unemployment': 0.01,  # 1% per year
    'Unemployment rate': 0.01,  # 1% per year
    'All-items': 0.01,  # 1% per year
    'Gasoline': 0.01,  # 1% per year
    'Goods': 0.01,  # 1% per year
    'Household operations, furnishings and equipment': 0.01,  # 1% per year
    'Shelter': 0.01,  # 1% per year
    'Transportation': 0.01,  # 1% per year
    'Emigrants': 0.01,  # 1% per year
    'Immigrants': 0.01,  # 1% per year
    'Net emigration': 0.01,  # 1% per year
    'Net non-permanent residents': 0.01,  # 1% per year
    'Net temporary emigration': 0.01,  # 1% per year
    'Returning emigrants': 0.01,  # 1% per year
    'Average income excluding zeros': 0.02,  # 2% per year
    'Median income excluding zeros': 0.015,  # 1.5% per year
}

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
        'Average income excluding zeros': [latest_data['Average income excluding zeros']],
        'Median income excluding zeros': [latest_data['Median income excluding zeros']],
        'HPI': [latest_data['HPI']]
    })

input_data = get_features(house_type, province, area)

# Predict button
if st.button('Predict Benchmark Value'):
    if input_data is not None:
        if average_increase_percentage is not None:
            try:
                predictions = []
                years_range = list(range(1, years + 1))
                
                for year in years_range:
                    # Adjust features for each year (e.g., applying growth factors)
                    input_data_adjusted = input_data.copy()
                    
                    # Apply percentage increases to all features
                    for feature, growth_rate in growth_rates.items():
                        if feature in input_data_adjusted.columns:
                            input_data_adjusted[feature] *= (1 + growth_rate * year)  # Apply growth rate (as a percentage)
                    
                    # Apply average increase percentage to HPI
                    if 'HPI' in input_data_adjusted.columns:
                        input_data_adjusted['HPI'] *= (1 + average_increase_percentage / 100 * year)  # Apply average increase (as a percentage)
                    
                    # Preprocess the adjusted input data
                    input_data_processed = preprocessor.transform(input_data_adjusted)
                    
                    # Make prediction for the adjusted data
                    prediction = model.predict(input_data_processed)[0]
                    predictions.append(prediction)
                
                # Plot predictions
                plt.figure(figsize=(10, 6))
                plt.plot(years_range, predictions, marker='o', linestyle='-', color='b')
                plt.title('Predicted Benchmark Value Over Years')
                plt.xlabel('Year')
                plt.ylabel('Predicted Benchmark Value')
                plt.grid(True)
                plt.xticks(years_range)
                plt.tight_layout()
                
                st.pyplot(plt)
            except Exception as e:
                st.error(f'Error making prediction: {e}')
        else:
            st.error('Error: Average increase percentage is None.')
    else:
        st.error('Error: Input data is None.')
