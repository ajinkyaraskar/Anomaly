# Changed display summary fn in create prompt and display summary

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from preprocess import CustomPreprocessor, MeanEncodingTransformer, ColumnSelector


import joblib
import os
import pickle
import joblib
from lime.lime_tabular import LimeTabularExplainer
import shap
import plotly.express as px
import matplotlib.pyplot as plt
from openai import AzureOpenAI

import dotenv
dotenv.load_dotenv()

deployment_name = os.environ['DEPLOYMENT_NAME']
api_key = os.environ["OPENAI_API_KEY"]
azure_endpoint = os.environ['AZURE_ENDPOINT']
api_version = os.environ['OPENAI_API_VERSION']

client = AzureOpenAI(
  api_key=api_key,  
  azure_endpoint=azure_endpoint,
  api_version=api_version
)

def get_completion(prompt, model=deployment_name):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content

major_categories_path = 'data/major_categories.pkl'
target_mean_dict_path = "data/target_mean_dict.pkl"
test_data_path = 'data/X_test_sample.csv'
train_data_path = 'data/X_train_sample.csv'
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
                'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt', 'OPAnnualDeductibleAmt']
                #  'PotentialFraud']

mean_enc_columns = ['AttendingPhysician','OperatingPhysician','OtherPhysician','ClmDiagnosisCode_1']

training_cols = ['InscClaimAmtReimbursed', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'ClmDiagnosisCode_1', 'DeductibleAmtPaid', 'IPD_OPD',
       'Gender', 'Race', 'RenalDiseaseIndicator', 'State', 'County', 'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure',
       'ChronicCond_KidneyDisease', 'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression', 'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart',
       'ChronicCond_Osteoporasis', 'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'IPAnnualReimbursementAmt', 'IPAnnualDeductibleAmt', 'OPAnnualReimbursementAmt',
       'OPAnnualDeductibleAmt', 'NoOfDays_Admitted', 'Age', 'WhetherDead']

# Define columns for imputation
columns_to_impute = ['DeductibleAmtPaid'] 
unimputed = [col for col in training_cols if col not in columns_to_impute]



def create_prompt(features, shap_values):
    explanation = "\n".join([f"{feature}: {value:.2f}" for feature, value in zip(features, shap_values)])
    
    prompt = f"""
    The following are the key factors that influence the prediction for the claim:

    1. The feature names and their respective contributions (SHAP values) are listed below:
    {explanation}

    2. The prediction is based on these factors:
        - Positive SHAP values indicate that the feature contributes to increasing the likelihood of fraud.
        - Negative SHAP values suggest that the feature is associated with a lower likelihood of fraud.
        - A value near zero means the feature has little to no impact on the prediction.

    In this case:
        - If a feature has a large positive value, it is pushing the model to classify the claim as fraudulent.
        - If a feature has a large negative value, it is pulling the model towards classifying the claim as non-fraudulent.

    Please focus on the features with the highest positive and negative SHAP values to understand what the model considers most important when making the decision.
    """
    
    return prompt

def create_prompt(features, shap_values, base_value, prediction, prediction_prob):
    # Normalize SHAP values to percentages for relative contributions
    total_shap = sum(abs(value) for value in shap_values)
    shap_percentages = [(value / total_shap) * 100 for value in shap_values]

    # Format the explanation with percentages
    explanation = "\n".join(
        [
            f"{feature}: {value:.2f}% ({'increases' if shap > 0 else 'decreases'} the likelihood of fraud)"
            for feature, value, shap in zip(features, shap_percentages, shap_values)
        ]
    )

    # Add clarity with prediction label and probabilities
    fraud_prob = prediction_prob[1] * 100  # Probability of fraud (class 1)
    non_fraud_prob = prediction_prob[0] * 100  # Probability of non-fraud (class 0)
    label_text = "Fraudulent" if prediction == 1 else "Not Fraudulent"

    # Improved prompt for LLM
    prompt = f"""
    The claim has been predicted as **{label_text}** with the following probabilities:
        - Fraudulent: {fraud_prob:.2f}%
        - Not Fraudulent: {non_fraud_prob:.2f}%

    This prediction is based on the following factors and their contributions (in percentage) to the likelihood of fraud detection:

    1. Baseline Prediction (Model's Default Probability without Feature Contributions): {base_value:.2f}
    2. Final Prediction (Including Feature Contributions): {fraud_prob:.2f}% for Fraudulent.

    Feature Contributions:
    {explanation}

    How to interpret this:
        - A positive percentage indicates that the feature increases the likelihood of fraud.
        - A negative percentage suggests that the feature decreases the likelihood of fraud.
        - Larger absolute percentages mean the feature has a more significant impact on the prediction.

    Please summarize the key factors influencing this prediction, focusing on the features with the highest contributions (positive or negative) and their significance to the final outcome.
    """
    return prompt

def create_prompt2(features, shap_values, prediction_label, prediction_prob):
    # Normalize SHAP values to percentages for relative contributions
    total_shap = sum(abs(value) for value in shap_values)
    shap_percentages = [(value / total_shap) * 100 for value in shap_values]

    # Format the explanation with percentages
    explanation = "\n".join(
        [
            f"{feature}: {value:.2f}% ({'increases' if shap > 0 else 'decreases'} the likelihood of fraud)"
            for feature, value, shap in zip(features, shap_percentages, shap_values)
        ]
    )

    # Use only fraud probability
    fraud_prob = prediction_prob * 100  # Probability of fraud

    # Improved prompt for LLM
    prompt = f"""
    The model predicts that the claim has a **{fraud_prob:.2f}% likelihood of being fraudulent**.

    This prediction is based on the following factors and their contributions (in percentage) to the likelihood of fraud:

    Feature Contributions:
    {explanation}

    How to interpret this:
        - A positive percentage indicates that the feature increases the likelihood of fraud.
        - A negative percentage suggests that the feature decreases the likelihood of fraud.
        - Larger absolute percentages mean the feature has a more significant impact on the prediction.

    Please summarize the key factors influencing this prediction, focusing on the features with the highest contributions (positive or negative) and their significance to the final outcome.
    """
    return prompt

# create_prompt
def create_prompt3(features, shap_values, prediction_label, prediction_prob):
    # Normalize SHAP values to percentages for relative contributions
    total_shap = sum(abs(value) for value in shap_values)
    shap_percentages = [(value / total_shap) * 100 for value in shap_values]

    # Format the feature contributions with percentages
    explanation = "\n".join(
        [
            f"{feature}: {value:.2f}% ({'increases' if shap > 0 else 'decreases'} the likelihood of fraud)"
            for feature, value, shap in zip(features, shap_percentages, shap_values)
        ]
    )

    # Add few-shot examples to guide LLM
    few_shot_examples = """
    Example 1:
    The model predicts that the claim is **Fraud** with a **75.00% likelihood of being fraudulent**.

    Key factors influencing this prediction:
    - County: 40.00% (increases the likelihood of fraud)
    - State: 20.00% (increases the likelihood of fraud)
    - Attending Physician: -10.00% (decreases the likelihood of fraud)

    In this case, the claim is flagged as fraudulent due to the high contribution of "County" and "State," which significantly increased the likelihood of fraud. On the other hand, the "Attending Physician" reduces the likelihood, but its impact is less significant.

    Example 2:
    The model predicts that the claim is **Not Fraud** with a **15.00% likelihood of being fraudulent**.

    Key factors influencing this prediction:
    - Deductible Amount Paid: -50.00% (decreases the likelihood of fraud)
    - IPD/OPD Indicator: -30.00% (decreases the likelihood of fraud)
    - Annual Reimbursement Amount: 10.00% (increases the likelihood of fraud)

    In this case, the model confidently predicts the claim as not fraudulent. Features like "Deductible Amount Paid" and "IPD/OPD Indicator" strongly reduce the likelihood of fraud, outweighing the minor positive contribution from "Annual Reimbursement Amount."
    """

    # Main prompt with disclaimer
    prompt = f"""
    The model predicts that the claim is **{prediction_label}** with a **{prediction_prob:.2f}% likelihood of being fraudulent**.

    This prediction is influenced by the following factors and their contributions (in percentage) to the likelihood of fraud:

    Feature Contributions:
    {explanation}

    How to interpret this:
        - A positive percentage indicates that the feature increases the likelihood of fraud.
        - A negative percentage suggests that the feature decreases the likelihood of fraud.
        - Larger absolute percentages mean the feature has a more significant impact on the prediction.

    {few_shot_examples}
    
    Use only top 5 contributors to decision while generating response and keep word count limited between 100 to 150 words.
    Disclaimer:
    These explanations are generated based on SHAP values, which are used to interpret the model's predictions. While SHAP provides insights into feature importance, these explanations are approximations and should be used as guidance rather than definitive reasoning.
    """
    return prompt



def display_explanations(prompt):
    st.write(prompt)
    # Call LLM explanation (assuming `get_completion` function exists)
    llm_explanation = get_completion(prompt)  # Replace with actual function to call LLM
    st.write("### LLM Explanation:")
    st.write(llm_explanation)

    # st.write("### LIME Explanation for Selected Claim")
    # st.pyplot(exp.as_pyplot_figure())

# Streamlit UI
st.title("Fraud Detection Dashboard")
# Input Data
input_data = pd.read_csv(test_data_path, usecols=required_cols)
st.write("Data Preview:", input_data.head())

# Load the pipeline
pipeline = joblib.load(pipeline_path)

try:
    # Predict using the pipeline
    predictions = pipeline.predict(input_data)
    # input_data["Fraud_Predicted"] = predictions
    input_data["Fraud_Predicted"] = np.where(predictions == 1, "Potential Fraud", "Not Fraud")
    input_data = input_data[['Fraud_Predicted'] + [col for col in input_data.columns if col != 'Fraud_Predicted']]

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
    # st.write(fraud_counts)
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

except Exception as e:
    st.error(f"An error occurred during prediction: {e}")

# Lime and LLm explanation 
feature_names = pipeline.named_steps["column_transformer"].get_feature_names_out()
feature_names = [name.split('__', 1)[-1] for name in feature_names]

X_test_transformed = pipeline[:-1].transform(input_data)  # Transform the data without the model
X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=feature_names)

# Create shap explainer and get shap values 
def get_shap_values(X_test_transformed_df, pipeline):
    explainer = shap.Explainer(pipeline[-1])
    shap_values = explainer(X_test_transformed_df)
    return shap_values

# Create a Streamlit button for showing the SHAP summary plot
# st.title("SHAP Feature Contribution Visualization")

shap_values = get_shap_values(X_test_transformed_df, pipeline)

# Button to show the SHAP summary plot
if st.button("Show Feature Contribution"):
    # Display the SHAP summary plot when the button is clicked
    # st.write("Displaying SHAP summary plot...")  # Optional: Uncomment this for additional text display
    fig = plt.figure()  # Create a new figure
    shap.summary_plot(shap_values, X_test_transformed_df, plot_type="bar") # Generate the SHAP plot
    st.pyplot(fig)

# Local Explanation 
# Input Claim ID
default_claim_id = str(input_data['ClaimID'].iloc[0])
selected_claim_id = st.text_input("Enter Claim ID:", value=default_claim_id)

if st.button("Generate Explanation"):
# if selected_claim_id:
    selected_case = input_data[input_data['ClaimID'] == selected_claim_id]
    if selected_case.empty:
        st.error(f"Claim ID {selected_claim_id} not found in the dataset.")
    else:
        idx = selected_case.index[0]
        data_row = input_data.iloc[[idx]]
        prediction = pipeline.predict(data_row)[0]
        prediction_label = "Fraud" if prediction == 1 else "Not Fraud"
        prediction_prob = pipeline.predict_proba(data_row)[0][1]
        # Preprocess data for LIME
        transformed_data_row = pipeline[:-1].transform(data_row)
        
        # Shap
        instance_shap_values = shap_values[idx]     
        if isinstance(instance_shap_values, shap.Explanation):
            instance_shap_values = instance_shap_values.values 

        d1 = {}

        for k,v in zip(X_test_transformed_df.columns,instance_shap_values):
            d1[k]=v

        st.write(d1)

        # prompt = create_prompt(prediction, prediction_prob)
        # prompt = create_prompt(X_test_transformed_df.columns, instance_shap_values)
        prompt = create_prompt3(X_test_transformed_df.columns, instance_shap_values, prediction_label, prediction_prob)

        
        # Display explanations
        display_explanations(prompt)
