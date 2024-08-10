import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import KNNImputer

# Load your pre-trained XGBoost model
# (Ensure you have trained and saved the model earlier)
model = xgb.XGBRegressor(colsample_bytree=1.0, learning_rate=0.1, max_depth=10, n_estimators=500, subsample=1.0)
model.load_model('xgboost_model.json')  # Load the model from a file

# Sample input data structure
# (In practice, replace with your dataset's structure)
areas = ["GREATER_VANCOUVER", "LOWER_MAINLAND", "OAKVILLE_MILTON","FRASER_VALLEY","MISSISSAUGA","GREATER_TORONTO"
         ,"VICTORIA","CHILLIWACK_AND_DISTRICT","HAMILTON_BURLINGTON","GUELPH_AND_DISTRICT","KITCHENER_WATERLOO",
         "BARRIE_AND_DISTRICT","CAMBRIDGE","INTERIOR_BC","LAKELANDS","VANCOUVER_ISALAND","BRANTFORD_REGION",
         "NORTHUMBERLAND_HILLS","WOODSTOCK_INGERSOLL","LONDON_ST_THOMAS","NIAGARA_REGION","KAWARTHA_LAKES",
         "PETERBOROUGH_AND_KAWARTHAS","WINDSOR_ESSEX","MONTREAL_CMA","CALGARY","RIDEAU_ST_LAWRENCE","QUINTE_AND_DISTRICT",
         "HURON_PERTH","SIMCOE_AND_DISTRICT","KINGSTON_AND_AREA","GREY_BRUCE_OWEN_SOUND","TILLSONBURG_DISTRICT",
         "BANCROFT_AND_AREA","HALIFAX_DARTMOUTH","SASKATOON","EDMONTON","SUDBURY","QUEBEC_CMA","ESTRIE","NORTH_BAY",
         "WINNIPEG","REGINA","GREATER_MONCTON","ST_JOHNS_NL","SAINT_JOHN_NB","SAULT_STE_MARIE","FREDERICTON","MAURICIE",
         "CENTRE_DU_QUEBEC"]
house_types = ["Apartment", "Composite", "Type3","One_Storey","Two_Storey","Single_Family","Townhouse"]
provinces = ["BC", "ON", "AB","MB","NB","NL","NS","QC","SK"]

# Set up Streamlit app
st.title("House Price Benchmark Predictor")

# Input section
house_type = st.selectbox("Select House Type", house_types)
province = st.selectbox("Select Province", provinces)
area = st.selectbox("Select Area", areas)
years = st.slider("Select Number of Years to Predict", 1, 10)

# Predict button
if st.button("Predict Benchmark Value"):
    # Example input features based on user input
    # Replace with your actual feature preparation
    input_data = pd.DataFrame({
        'House_Type': [house_type],
        'Province': [province],
        'Area': [area],
        'Date_Id': [datetime.now().year + i for i in range(years)]
    })

    # Example transformation steps (replace with your actual preprocessing)
    imputer = KNNImputer(n_neighbors=5)
    scaler = QuantileTransformer(output_distribution='normal')
    input_data = imputer.fit_transform(input_data)
    input_data = scaler.fit_transform(input_data)

    # Make predictions
    predictions = model.predict(input_data)

    # Display predictions
    st.write(f"Predicted Benchmark Values for the next {years} years:")
    for i, value in enumerate(predictions, 1):
        st.write(f"Year {datetime.now().year + i}: {value:.2f}")

    # Optional: Plot the predictions over time
    st.line_chart(predictions)

# Footer
st.write("© 2024 House Price Benchmark Predictor")