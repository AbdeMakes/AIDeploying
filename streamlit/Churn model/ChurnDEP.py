import streamlit as st
import numpy as np
import joblib as jb

# Load the Random Forest model
model = jb.load('churnfinal/Churn_mod.pkl')

# Sidebar title
page = st.sidebar.title('Customer Churn Prediction')

# Function to make predictions
def predict(input_features):
    prediction = model.predict(input_features)
    return prediction

# Function to encode categorical features
def encode_features(input_data):
    # Encoding mappings (based on how the model was trained)
    mappings = {
        'gender': {'Female': 0, 'Male': 1},
        'SeniorCitizen': {'No': 0, 'Yes': 1},
        'Partner': {'No': 0, 'Yes': 1},
        'Dependents': {'No': 0, 'Yes': 1},
        'PhoneService': {'No': 0, 'Yes': 1},
        'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'No': 0, 'Yes': 1},
        'PaymentMethod': {'Electronic check': 2, 'Mailed check': 3, 'Bank transfer (automatic)': 0, 'Credit card (automatic)': 1}
    }
    
    for feature in mappings:
        input_data[feature] = mappings[feature][input_data[feature]]
    
    return input_data

def main():
    st.title('Customer Churn Prediction')
    st.write('Enter the features below to get a churn prediction:')

    # Input fields for features
    gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ['No', 'Yes'])
    Partner = st.sidebar.selectbox('Partner', ['No', 'Yes'])
    Dependents = st.sidebar.selectbox('Dependents', ['No', 'Yes'])
    PhoneService = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    OnlineBackup = st.sidebar.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    DeviceProtection = st.sidebar.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    StreamingTV = st.sidebar.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ['No', 'Yes'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    tenure = st.sidebar.number_input('Tenure (months)', min_value=0, max_value=72, value=0)
    MonthlyCharges = st.sidebar.number_input('Monthly Charges', min_value=0.0, value=0.0)
    TotalCharges = st.sidebar.number_input('Total Charges', min_value=0.0, value=0.0)

    # Collect input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Encode the input data
    encoded_data = encode_features(input_data)

    # Prepare the input for prediction
    input_features = np.array([list(encoded_data.values())])

    if st.button('Predict'):
        prediction = predict(input_features)
        st.write('Prediction:', 'Churn' if prediction[0] else 'No Churn')

if __name__ == '__main__':
    main()
