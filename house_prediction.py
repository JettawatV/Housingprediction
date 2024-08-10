import streamlit as st
import pandas as pd
import zipfile
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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

# Verify if the loaded object is a valid model
if not hasattr(model, 'predict'):
    st.error('Loaded object is not a valid scikit-learn model.')
else:
    st.success('Model loaded successfully.')

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
        'HPI': [latest_data['HPI']]
    })

input_data = get_features(house_type, province, area)

# Predict button
if st.button('Predict Benchmark Value'):
    if input_data is not None:
        if average_increase_percentage is not None:
            try:
                predictions = []
                for year in range(1, years + 1):
                    # Adjust features for each year (e.g., applying growth factors)
                    input_data_adjusted = input_data.copy()
                    
                    # Apply percentage increases
                    input_data_adjusted['Population'] *= (1 + 0.01 * year)  # 1% growth per year
                    input_data_adjusted['Average income excluding zeros'] *= (1 + 0.02 * year)  # 2% growth per year
                    input_data_adjusted['Median income excluding zeros'] *= (1 + 0.015 * year)  # 1.5% growth per year
                    input_data_adjusted['HPI'] *= (1 + average_increase_percentage / 100 * year)  # Apply average increase (as a percentage)
                    
                    # Debug output
                    st.write(f'Year {year} Adjusted Data:')
                    st.write(input_data_adjusted)
                    
                    # Preprocess the adjusted input data
                    input_data_processed = preprocessor.transform(input_data_adjusted)
                    
                    # Debug output
                    st.write(f'Processed Data for Year {year}:')
                    st.write(input_data_processed)
                    
                    # Make prediction for the adjusted data
                    prediction = model.predict(input_data_processed)[0]
                    
                    predictions.append(prediction)
                
                # Display predictions for each year
                for i, prediction in enumerate(predictions, 1):
                    st.write(f'Predicted Benchmark Value for year {i}: {prediction:.2f}')
            except Exception as e:
                st.error(f'Error making prediction: {e}')
        else:
            st.error('Error: Average increase percentage is None.')
    else:
        st.error('Error: Input data is None.')
