import streamlit as st
import pandas as pd
import pickle
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
st.title('Housing Prices Prediction Based on HPI and Other Features')

# Input fields
house_type = st.selectbox('Select House Type', dataset['House_Type'].unique())
province = st.selectbox('Select Province', dataset['Province'].unique())
area = st.selectbox('Select Area', dataset['Area'].unique())

# Slider for number of years
years = st.slider('Select Number of Years to Predict', min_value=1, max_value=10, value=1)

# Get average increase for HPI
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

# Get the most recent data for all numerical features
def get_latest_features(house_type, province, area):
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
    return latest_data

latest_features = get_latest_features(house_type, province, area)

# Define numerical features and their growth percentages
numerical_features = [
    "HPI", "Net temporary emigration", "Net emigration", "Emigrants", "Shelter",
    "Unemployment", "Population", "Labour force", "Employment", "Average income excluding zeros"
]
growth_percentages = {
    "Net temporary emigration": 0.5,
    "Net emigration": 0.5,
    "Emigrants": 0.5,
    "Shelter": 0.5,
    "Unemployment": 0.5,
    "Population": 0.5,
    "Labour force": 0.5,
    "Employment": 0.5,
    "Average income excluding zeros": 0.5
}

# Predict button
if st.button('Predict Housing Price Value'):
    if latest_features is not None:
        if average_increase_percentage is not None:
            try:
                # Prepare the input data
                input_data = pd.DataFrame([latest_features], columns=numerical_features)
                
                # Initialize a list for predictions
                predictions = []
                years_range = list(range(1, years + 1))
                
                for year in years_range:
                    adjusted_features = input_data.copy()
                    
                    # Apply growth percentage for each feature
                    for feature in numerical_features:
                        if feature != "HPI":
                            growth_rate = growth_percentages.get(feature, 0)
                            adjusted_features[feature] *= (1 + growth_rate / 100 * year)
                        else:
                            # Adjust HPI based on the average increase
                            adjusted_hpi = adjusted_features["HPI"] * (1 + average_increase_percentage / 100 * year)
                            adjusted_features["HPI"] = adjusted_hpi
                    
                    # Preprocess the data
                    preprocessor = StandardScaler()
                    X_preprocessed = preprocessor.fit_transform(adjusted_features)
                    
                    # Make prediction for the adjusted data
                    prediction = model.predict(X_preprocessed)[0]
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
        st.error('Error: Latest features are None.')
