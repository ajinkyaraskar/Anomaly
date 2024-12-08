# adding project structure to 4.2

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from preprocess import CustomPreprocessor, MeanEncodingTransformer, ColumnSelector

import shap
import joblib
import pickle
import yaml

import plotly.express as px
import matplotlib.pyplot as plt
from utils.llm_utils import *

# Load YAML config
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access variables from the YAML file
major_categories_path = config['paths']['major_categories']
target_mean_dict_path = config['paths']['target_mean_dict']
test_data_path = config['paths']['test_data']
train_data_path = config['paths']['train_data']
pipeline_path = config['paths']['pipeline']

target_column = config['columns']['target']
required_cols = config['columns']['required']
mean_enc_columns = config['columns']['mean_encoding']
training_cols = config['columns']['training']
columns_to_impute = config['columns']['to_impute']

# Define columns for imputation
unimputed = [col for col in training_cols if col not in columns_to_impute]

def display_explanations(prompt):
    llm_explanation = get_completion(prompt)  # Replace with actual function to call LLM
    st.write("### LLM Explanation:")
    st.write(llm_explanation)

    if st.button("Show Prompt"):
        st.write(prompt)

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
