import streamlit as st 
import pickle
import pandas as pd

model = pickle.load(open("RandomForestRegressor.pkl", "rb"))


# Define the function for making predictions
def predict_selling_price(features):
    # Create a dataframe with the input features
    df = pd.DataFrame(features, index=[0])
    # Make the prediction using the pre-trained model
    prediction = model.predict(df)[0]
    return prediction*100000

# Define the Streamlit app
def main():
    # Set the title and description of the app
    st.title('Car Selling Price Prediction')
    st.write('This app predicts the selling price of a car based on its features.')
    
    # Create input fields for the car features
    present_price = st.number_input('Present Price in Indian Rupees', min_value=0, max_value=5000000, step=10000)
    kms_driven = st.number_input('Kilometers Driven', min_value=0, max_value=1000000, step=1000)
    owner = st.selectbox('Number of Previous Owners', [0, 1, 2, 3])
    age = st.number_input('Age of Car (in years)', min_value=0, max_value=50, step=1)
    fuel_type_diesel = st.checkbox('Fuel Type: Diesel')
    fuel_type_petrol = st.checkbox('Fuel Type: Petrol')
    seller_type_individual = st.checkbox('Seller Type: Individual')
    transmission_manual = st.checkbox('Transmission: Manual')
    
    # Create a button for making the prediction
    if st.button('Predict Selling Price'):
        # Create a dictionary with the input features
        features = {'Present_Price': present_price/100000,
                    'Kms_Driven': kms_driven,
                    'Owner': owner,
                    'Age': age,
                    'Fuel_Type_Diesel': fuel_type_diesel,
                    'Fuel_Type_Petrol': fuel_type_petrol,
                    'Seller_Type_Individual': seller_type_individual,
                    'Transmission_Manual': transmission_manual}
        # Make the prediction using the predict_selling_price function
        prediction = predict_selling_price(features)
        # Display the predicted selling price
        st.write(f'The predicted selling price is {prediction: .2f} .')

# Run the Streamlit app
if __name__ == '__main__':
    main()
