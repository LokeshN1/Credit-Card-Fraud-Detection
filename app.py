import streamlit as st
import numpy as np
import joblib

# Apply custom CSS for a dark mode theme
st.markdown(
    """
    <style>
    .main {
        background-color: #121212;
        padding: 20px;
        color: #e0e0e0;
    }
    .title {
        color: #bb86fc;
        text-align: center;
        font-family: 'Helvetica', sans-serif;
        font-size: 36px;
        margin-bottom: 20px;
    }
    .stRadio > label {
        font-size: 18px;
        color: #e0e0e0;
    }
    .stTextArea label, .stNumberInput label, .stTextInput label {
        font-size: 16px;
        color: #e0e0e0;
    }
    .stTextArea textarea, .stNumberInput input, .stTextInput input {
        background-color: #333;
        border: 1px solid #444;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: #e0e0e0;
        margin-bottom: 10px;
    }
    .stButton > button {
        background-color: #bb86fc;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        margin-top: 20px;
    }
    .stAlert {
        border-radius: 10px;
        padding: 15px;
        font-size: 16px;
        color: #e0e0e0;
        background-color: #333;
    }
    .footer {
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #888;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown("<h1 class='title'>Credit Card Fraud Detector</h1>", unsafe_allow_html=True)

# Load the saved model and scaler
model = joblib.load("credit_card_model.pkl")
scaler = joblib.load("scaler.pkl")

# User input form
st.markdown("### Enter Transaction Details for Prediction")

# Select input method
input_method = st.radio("Choose input method", ["Bulk input", "Individual fields"])

# Define the features based on your training data (excluding the target 'Class')
feature_names = [
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
    'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
    'V26', 'V27', 'V28', 'Amount'
]

# Initialize user_input list
user_input = [0.0] * len(feature_names)

if input_method == "Bulk input":
    st.markdown("Enter the transaction details as a comma-separated list of values:")
    bulk_input = st.text_area("Bulk input", "")

    if bulk_input:
        try:
            user_input_list = [float(i) for i in bulk_input.split(",")]
            if len(user_input_list) == len(feature_names):
                user_input = user_input_list  # Update the user_input list
            else:
                st.error("The number of input values does not match the number of features.")
        except ValueError:
            st.error("Please enter valid numeric values separated by commas.")

# Populate individual fields with values from user_input
cols = st.columns(3)
for idx, feature in enumerate(feature_names):
    with cols[idx % 3]:
        user_input[idx] = st.number_input(f"Enter value for {feature}", value=float(user_input[idx]))

# Prediction button
if st.button("Predict"):
    user_input_array = np.array(user_input).reshape(1, -1)
    
    # Scale only the 'Amount' feature
    amount_index = feature_names.index('Amount')
    user_input_array[0, amount_index] = scaler.transform([[user_input_array[0, amount_index]]])
    
    prediction = model.predict(user_input_array)

    if prediction[0] == 0:
        st.success("Prediction: Normal Transaction")
    else:
        st.error("Prediction: Fraud Transaction")

# Footer
st.markdown("<div class='footer'>© 2024 Credit Card Fraud Detection</div>", unsafe_allow_html=True)
