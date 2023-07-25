import streamlit as st
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Set page title and layout
st.set_page_config(page_title="CAR CO2 PREDICTION", layout="wide")

# Display app title
st.markdown("# CAR CO2 PREDICTION")

# Create form
with st.form("CO2 Prediction Form"):
    # Define form fields
    engine_size = st.number_input("Engine Size", min_value=0.0, step=0.1)
    cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12, 16])
    transmission = st.selectbox("Transmission", ["AS5", "M6", "AV7", "AS6", "AM6", "A6", "AM7", "AV8", "AS8", "A7", "A8", "M7", "A4", "M5", "AV", "A5", "AS7", "A9", "AS9", "AV6", "AS4", "AM5", "AM8", "AM9", "AS10", "A10", "AV10"])
    fuel_type = st.selectbox("Fuel Type", ["Z", "D", "X", "E", "N"])
    fuel_consumption_city = st.number_input("Fuel Consumption City", min_value=0.0, step=0.1)
    fuel_consumption_hwy = st.number_input("Fuel Consumption Highway", min_value=0.0, step=0.1)
    fuel_consumption_comb = st.number_input("Fuel Consumption Combined", min_value=0.0, step=0.1)
    fuel_consumption_comb_mpg = st.number_input("Fuel Consumption Combined MPG", min_value=0)
    make_type = st.selectbox("Make Type", ["Luxury", "Premium", "Sports", "General"])
    vehicle_class_type = st.selectbox("Vehicle Class Type", ["Hatchback", "SUV", "Sedan", "Truck"])

    # Create submit button
    submit_button = st.form_submit_button(label="Predict CO2 Emissions")

# Define function to predict CO2 emissions
def predict_co2_emissions():
    # Create CustomData object with form values
    data = CustomData(
        engine_size=engine_size,
        cylinders=cylinders,
        transmission=transmission,
        fuel_type=fuel_type,
        fuel_consumption_city=fuel_consumption_city,
        fuel_consumption_hwy=fuel_consumption_hwy,
        fuel_consumption_comb=fuel_consumption_comb,
        fuel_consumption_comb_mpg=fuel_consumption_comb_mpg,
        make_type=make_type,
        vehicle_class_type=vehicle_class_type
    )

    # Get prediction from pipeline
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(data.get_data_as_data_frame())

    # Display prediction result
    st.write(f"The CO2 prediction is: {results[0]}")

# Call predict_co2_emissions function when submit button is clicked
if submit_button:
    predict_co2_emissions()
