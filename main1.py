# pip install streamlit pandas numpy xgboost lime scikit-learn openai

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import pickle
import joblib
import os
import streamlit as st
import pandas as pd
import pickle
import joblib
import xgboost as xgb
# import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer


# from openai import Completion  # For LLM
import numpy as np

from openai import AzureOpenAI

import dotenv
dotenv.load_dotenv()

from preprocess import CustomPreprocessor, MeanEncodingTransformer, ColumnSelector

major_categories_path = 'data/major_categories.pkl'
target_mean_dict_path = "data/target_mean_dict.pkl"
test_data_path = 'data/sample_data.csv'
target_column  = 'PotentialFraud'
pipeline_path = "fraud_detection_pipeline.pkl"

required_cols = ['BeneID', 'ClaimID', 'ClaimStartDt', 'ClaimEndDt', 'Provider', 'InscClaimAmtReimbursed',
                'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1',
                'DeductibleAmtPaid', 'IPD_OPD', 'AdmissionDt', 'DischargeDt', 'DOB', 'DOD',
                'Gender', 'Race', 'RenalDiseaseIndicator', 'State', 'County',
                'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
                'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
                'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
                'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'IPAnnualReimbursementAmt',
                'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt',
                 'PotentialFraud']

mean_enc_columns = ['AttendingPhysician','OperatingPhysician','OtherPhysician','ClmDiagnosisCode_1']

training_cols = ['InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1', 'DeductibleAmtPaid', 'IPD_OPD',
       'Gender', 'Race', 'RenalDiseaseIndicator', 'State', 'County', 'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
       'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
       'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt', 'NoOfDays_Admitted', 'Age', 'WhetherDead']

# Define columns for imputation
columns_to_impute = ['DeductibleAmtPaid'] 
unimputed = [col for col in training_cols if col not in columns_to_impute]



import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# Streamlit App
st.title("Fraud Detection Dashboard")

# Input Data
input_data = pd.read_csv(test_data_path, usecols=required_cols)
st.write("Uploaded Data Preview:", input_data.head())

# Load the pipeline
pipeline = joblib.load(pipeline_path)

try:
    # Predict using the pipeline
    predictions = pipeline.predict(input_data)
    input_data["Fraud_Predicted"] = predictions
    
    # Calculate statistics
    total_cases = len(input_data)
    potential_fraud_cases = sum(predictions)
    non_fraud_cases = total_cases - potential_fraud_cases
    
    # Display tiles at the top
    st.write("### Summary Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases", total_cases)
    col2.metric("Potential Fraud Cases", potential_fraud_cases)
    col3.metric("Not Fraud Cases", non_fraud_cases)
    
    # Display pie chart before the predictions DataFrame
    st.write("### Fraud vs Not Fraud Cases")
    fraud_counts = input_data["Fraud_Predicted"].value_counts()
    pie_chart = px.pie(
        values=fraud_counts.values,
        names=["Not Fraud", "Potential Fraud"],
        title="Fraud vs Not Fraud Cases",
        color_discrete_sequence=["green", "red"]
    )
    st.plotly_chart(pie_chart)
    
    # Display predictions after the pie chart
    st.write("### Predictions")
    st.write(input_data)
    
    # Optional Additional Visuals
    if "timestamp" in input_data.columns:  # Replace with your timestamp column name
        st.write("### Fraud Rate Over Time")
        input_data["timestamp"] = pd.to_datetime(input_data["timestamp"])
        fraud_rate = input_data.groupby(input_data["timestamp"].dt.date)["Fraud_Predicted"].mean()
        fraud_rate_fig = px.line(
            fraud_rate,
            title="Fraud Rate Over Time",
            labels={"value": "Fraud Rate", "index": "Date"},
        )
        st.plotly_chart(fraud_rate_fig)
    
    if "category_column" in input_data.columns:  # Replace with your categorical column
        st.write("### Fraud Cases Across Categories")
        category_fraud = input_data.groupby("category_column")["Fraud_Predicted"].sum()
        category_chart = px.bar(
            category_fraud,
            title="Fraud Cases Across Categories",
            labels={"value": "Fraud Cases", "index": "Category"},
        )
        st.plotly_chart(category_chart)

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")


# Model Predictions
# predictions = model.predict(preprocessed_data)
# prediction_probs = model.predict_proba(preprocessed_data)
# test_data["Prediction"] = predictions
# test_data["Fraud Probability"] = prediction_probs[:, 1]
# st.write("Predictions:", test_data)


# Preprocess input data for LIME initialization
preprocessed_data = pipeline[:-1].transform(input_data)  # Apply pipeline steps up to classifier
feature_names = pipeline.named_steps["column_transformer"].get_feature_names_out()

# Initialize LimeTabularExplainer
explainer = LimeTabularExplainer(
    preprocessed_data,
    feature_names=feature_names,
    class_names=["Not Fraud", "Potential Fraud"],
    discretize_continuous=True
)

def lime_explainer_with_llm(idx):
    # Select raw instance
    raw_instance = input_data.iloc[idx]
    
    # Preprocess the instance using the pipeline (excluding classifier)
    preprocessed_instance = pipeline[:-1].transform(raw_instance.values.reshape(1, -1))
    
    # Generate explanation
    explanation = explainer.explain_instance(
        preprocessed_instance.flatten(),
        pipeline.predict_proba,  # Use pipeline for prediction
        num_features=5
    )
    
    # Extract LIME explanation
    explanation_list = explanation.as_list()

    # Use LLM for summary
    llm_prompt = (
        "Given the following LIME explanation for a fraud detection model, "
        "provide a detailed summary of the case: \n\n"
        f"Instance details: {dict(zip(raw_instance.index, raw_instance.values))}\n\n"
        "LIME Explanation (features and their impact on prediction):\n"
        f"{explanation_list}\n\n"
        "Summarize the case and its likelihood of fraud in a clear, concise way."
    )
    llm_summary = llm(llm_prompt)
    
    # Display in Streamlit
    st.write("### Explanation for Selected Instance")
    st.write(f"**Claim ID**: {raw_instance['ClaimID']}")
    st.pyplot(explanation.as_pyplot_figure())
    st.write("### LLM Summary")
    st.write(llm_summary)


# Case Selection

# Streamlit UI for selecting a case
st.write("### Select a Case for Explanation by Claim ID")
# Case selection based on Claim ID
claim_ids = input_data["ClaimID"].unique()  # Replace 'ClaimID' with the correct column name
selected_claim_id = st.selectbox("Select Claim ID:", claim_ids)

# Filter data for the selected Claim ID
selected_case = input_data[input_data["ClaimID"] == selected_claim_id]

# Display selected case details
st.write("### Selected Case Details")
st.write(selected_case)
if st.button("Generate Explanation"):
    lime_explainer_with_llm(instance_idx)
