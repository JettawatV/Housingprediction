import streamlit as st
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Function to load the model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model('hpi_model.pkl')

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
                    
                    # Create a DataFrame with the specified feature values
                    input_data_adjusted = pd.DataFrame({
                        'HPI': [adjusted_hpi],
                        'Net temporary emigration': [default_values["Net temporary emigration"]],
                        'Net emigration': [default_values["Net emigration"]],
                        'Emigrants': [default_values["Emigrants"]],
                        'Shelter': [default_values["Shelter"]],
                        'Unemployment': [default_values["Unemployment"]],
                        'Population': [default_values["Population"]],
                        'Labour force': [default_values["Labour force"]],
                        'Employment': [default_values["Employment"]],
                        'Average income excluding zeros': [default_values["Average income excluding zeros"]]
                    })
                    
                    # Make prediction for the adjusted data
                    prediction = model.predict(input_data_adjusted)[0]
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
        st.error('Error: Latest HPI is None.')
