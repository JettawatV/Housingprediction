import streamlit as st
import pandas as pd
import zipfile
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.express as px

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
st.title('Housing Prices Prediction Based on HPI')

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

# Get the most recent HPI data
def get_latest_hpi(house_type, province, area):
    filtered_data = dataset[
        (dataset['House_Type'] == house_type) &
        (dataset['Province'] == province) &
        (dataset['Area'] == area)
    ]
    
    if filtered_data.empty:
        st.error('No data available for the selected inputs.')
        return None
    
    # Get the most recent data (or handle it accordingly)
    latest_data = filtered_data.iloc[-1]
    return latest_data['HPI']

latest_hpi = get_latest_hpi(house_type, province, area)

# Predict button
if st.button('Predict Housing Price Value'):
    if latest_hpi is not None:
        if average_increase_percentage is not None:
            try:
                predictions = []
                years_range = list(range(1, years + 1))
                
                for year in years_range:
                    # Adjust HPI for each year
                    adjusted_hpi = latest_hpi * (1 + average_increase_percentage / 100 * year)
                    
                    # Create DataFrame for prediction
                    input_data_adjusted = pd.DataFrame({
                        'HPI': [adjusted_hpi]
                    })
                    
                    # Preprocess the adjusted input data
                    input_data_processed = preprocessor.transform(input_data_adjusted)
                    
                    # Make prediction for the adjusted data
                    prediction = model.predict(input_data_processed)[0]
                    predictions.append(prediction)
                
                # Plot predictions
                df_plot = pd.DataFrame({
                    'Year': years_range,
                    'Predicted Benchmark Value': predictions
                })
                
                fig = px.line(df_plot, x='Year', y='Predicted Benchmark Value', markers=True, title='Predicted Benchmark Value Over Years')
                fig.update_traces(
                    mode='lines+markers+text',
                    text=[f'{value:.2f}' for value in df_plot['Predicted Benchmark Value']],
                    textposition='top center',
                    marker=dict(size=8, color='blue')
                )
                fig.update_layout(
                    xaxis_title='Year',
                    yaxis_title='Predicted Benchmark Value',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f'Error making prediction: {e}')
        else:
            st.error('Error: Average increase percentage is None.')
    else:
        st.error('Error: Latest HPI data is None.')
