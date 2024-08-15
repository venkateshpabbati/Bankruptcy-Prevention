import streamlit as st
import joblib
import numpy as np
import os

# Define paths to the models
MODEL_PATHS = {
    'Logistic Regression': {
        'GridSearchCV': 'best_logistic_regression_model.pkl',
        'RandomizedCV': 'best_logistic_regression_randomcv_model.pkl'
    },
    'SVM': {
        'GridSearchCV': 'best_svm_model.pkl',
        'RandomizedCV': 'best_svm_randomcv_model.pkl'
    }
}

# Function to load model
def load_model(model_name, search_type):
    model_path = MODEL_PATHS[model_name][search_type]
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None
    return joblib.load(model_path)

# Streamlit app
st.title("Bankruptcy Prediction")

# Dropdown for model selection
model_name = st.selectbox("Select Model", options=['Logistic Regression', 'SVM'])
search_type = st.selectbox("Select Search Type", options=['GridSearchCV', 'RandomizedCV'])

# Load the selected model
model = load_model(model_name, search_type)

if model is not None:
    # Helper function to get the risk description
    def get_risk_description(value):
        if value == 0:
            return "Low Risk"
        elif value == 0.5:
            return "Medium Risk"
        else:
            return "High Risk"

    # Create columns for sliders and their descriptions
    col1, col2 = st.columns(2)
    with col1:
        industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.0, 0.5)
        st.write(f"**Description**: {get_risk_description(industrial_risk)}")

        management_risk = st.slider("Management Risk", 0.0, 1.0, 0.0, 0.5)
        st.write(f"**Description**: {get_risk_description(management_risk)}")

        financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.0, 0.5)
        st.write(f"**Description**: {get_risk_description(financial_flexibility)}")

    with col2:
        credibility = st.slider("Credibility", 0.0, 1.0, 0.0, 0.5)
        st.write(f"**Description**: {get_risk_description(credibility)}")

        competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.0, 0.5)
        st.write(f"**Description**: {get_risk_description(competitiveness)}")

        operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.0, 0.5)
        st.write(f"**Description**: {get_risk_description(operating_risk)}")

    # Hardcode the missing feature value (e.g., 0)
    missing_feature = 0.0  # Change to 1.0 if needed

    # Convert input values to numpy array including missing feature
    input_features = np.array([[industrial_risk, management_risk, financial_flexibility,
                                credibility, competitiveness, operating_risk, missing_feature]])

    # Make predictions
    if st.button("Check Results"):
        # Make prediction using the loaded model
        prediction = model.predict(input_features)
        prediction_proba = model.predict_proba(input_features)

        result = "Bankruptcy" if prediction[0] == 1 else "Non-Bankruptcy"
        st.write(f"The model predicts: **{result}**")
        st.write(f"Probability of Non-Bankruptcy (0): **{prediction_proba[0][0]:.2f}**")
        st.write(f"Probability of Bankruptcy (1): **{prediction_proba[0][1]:.2f}**")
else:
    st.write("Please ensure that the model files are available and the paths are correct.")
