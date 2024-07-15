import streamlit as st
import numpy as np

import joblib as jb

model = jb.load('RFTools/predictive_maintenance.pkl')

page=st.sidebar.title('Prediction')

def predict(input_features):
    prediction = model.predict(input_features)
    return prediction

def encode_type(value):
    encoding = {'H': 0, 'L': 1, 'M': 2}
    return encoding[value]

def main():
    st.title('Your Machine Learning Model Deployment')
    st.write('Enter the features below to get predictions:')

    type_selector = st.sidebar.selectbox('Type Selector', ['L', 'M', 'H'])

    air_temperature = st.sidebar.number_input('Air temperature [K]')
    process_temperature = st.sidebar.number_input('Process temperature [K]')
    rotational_speed = st.sidebar.number_input('Rotational speed [rpm]')
    torque = st.sidebar.number_input('Torque [Nm]')
    tool_wear = st.sidebar.number_input('Tool wear [min]')

    encoded=encode_type(type_selector)
    input_data = np.array([[encoded, air_temperature, process_temperature, rotational_speed, torque, tool_wear]])

    if st.button('Predict'):
        prediction = predict(input_data)
        st.write('Prediction:', prediction)

if __name__ == '__main__':
    main()
