import streamlit as st
import pickle
import pandas as pd
import numpy as np

#importing the model
model = pickle.load(open("KNN_Classifier.pkl", "rb"))

# define the input fields for the user
st.sidebar.title("Diabetes Prediction")
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, max_value=17, value=1, step=1)
glucose = st.sidebar.number_input("Glucose Level", min_value=0, max_value=200, value=100, step=1)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=72, step=1)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=23, step=1)
insulin = st.sidebar.number_input("Insulin Level", min_value=0, max_value=846, value=30, step=1)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=67.1, value=32.0, step=1.0)
diabetes_pedigree_function = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.078, max_value=2.42, value=0.3725, step=0.01)
age = st.sidebar.number_input("Age", min_value=21, max_value=81, value=29)


# create a feature vector from the user inputs
features = {
    "Pregnancies": int(pregnancies),
    "Glucose": float(glucose),
    "BloodPressure": float(blood_pressure),
    "SkinThickness": float(skin_thickness),
    "Insulin": float(insulin),
    "BMI": float(bmi),
    "DiabetesPedigreeFunction": float(diabetes_pedigree_function),
    "Age": int(age),
    
   
}


# make a prediction using the model
def predict_outcome(features):
    # Create a dataframe with the input features
    df = pd.DataFrame(features, index=[0])
    # Make the prediction using the pre-trained model
    prediction = model.predict(df)[0]

    return prediction


prediction = predict_outcome(features)#finding the prediction


# display the prediction to the user
st.write("")
st.write("## Prediction")
if prediction == 0:
    st.write("No diabetes")
else:
    st.write("Diabetes detected")
