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

# Load your dataset (update with your dataset path)
dataset = pd.read_csv('housing.csv')  # Replace with the correct path to your dataset

# Load the preprocessor and model
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

model = load_zip_pipeline('xgboost_pipeline.zip', 'xgboost_pipeline.pkl')

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

# Slider for number of years to predict into the future
years = st.slider('Select number of years to predict into the future', 1, 10, 1)

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
        'Returning emigrants': [latest_data['Returning emigrants']]
    })

input_data = get_features(house_type, province, area)

# Predict button
if st.button('Predict Benchmark Value'):
    if input_data is not None:
        try:
            # Check if preprocessor is already fitted
            if not hasattr(preprocessor, 'transformers_'):
                st.error('Preprocessor is not fitted yet. Please fit the preprocessor before using it.')
            else:
                # Initialize an empty list to store predictions
                future_predictions = []
                
                # Make predictions for each year in the future
                for year in range(1, years + 1):
                    # Adjust features for each year (e.g., applying growth factors)
                    input_data_adjusted = input_data.copy()
                    input_data_adjusted['Population'] *= (1 + 0.01 * year)  # 1% growth per year
                    input_data_adjusted['Average income excluding zeros'] *= (1 + 0.02 * year)  # 2% growth per year
                    input_data_adjusted['Median income excluding zeros'] *= (1 + 0.015 * year)  # 1.5% growth per year
                    
                    # Preprocess the input data
                    input_data_processed = preprocessor.transform(input_data_adjusted)
                    
                    # Make prediction
                    prediction = model.predict(input_data_processed)
                    future_predictions.append(prediction[0])

                # Display predictions
                for i, prediction in enumerate(future_predictions):
                    st.write(f'Predicted Benchmark Value for year {i+1}: {prediction}')
                    
        except Exception as e:
            st.error(f'Error making prediction: {e}')
    else:
        st.error('Error: Input data is None.')
