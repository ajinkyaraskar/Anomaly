# pip install streamlit pandas numpy xgboost lime scikit-learn openai
import streamlit as st
import pandas as pd
import pickle
import xgboost as xgb
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
from openai import Completion  # For LLM
import numpy as np

print('-'*50,'imported')
# Load the XGBoost model
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load a sample dataset structure (to ensure preprocessing compatibility)
sample_data = pd.read_csv("sample_data.csv")  # This should contain column names and preprocessing logic

# Function for LLM integration
def get_completion(prompt):
    # Add your API key and logic for GPT-like models
    return Completion.create(prompt=prompt, engine="text-davinci-003", max_tokens=150)["choices"][0]["text"]

# Preprocessing function
def preprocess_data(df, sample_df):
    df = df.copy()
    # Ensure columns match the sample dataset
    df = df[sample_df.columns]
    # Standardize data (if applicable)
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled


# Streamlit App
st.title("Fraud Detection Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload your test data (CSV)", type=["csv"])
if uploaded_file:
    test_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", test_data.head())

    # Preprocess the data
    try:
        preprocessed_data = preprocess_data(test_data, sample_data)
        st.success("Data preprocessed successfully!")
    except Exception as e:
        st.error(f"Preprocessing error: {e}")

    # Model Predictions
    predictions = model.predict(preprocessed_data)
    prediction_probs = model.predict_proba(preprocessed_data)
    test_data["Prediction"] = predictions
    test_data["Fraud Probability"] = prediction_probs[:, 1]
    st.write("Predictions:", test_data)

    # Case Selection
    st.subheader("Select a case to explain")
    selected_index = st.number_input(
        "Enter the row index of the case you want to explain:",
        min_value=0,
        max_value=len(test_data) - 1,
        step=1
    )
    
    if st.button("Explain Selected Case"):
        selected_case = preprocessed_data[selected_index].reshape(1, -1)
        selected_row = test_data.iloc[selected_index]

        # LIME Explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(
            preprocessed_data,
            feature_names=test_data.columns[:-2],  # Exclude Prediction and Fraud Probability
            class_names=["Not Fraud", "Fraud"],
            mode="classification"
        )
        explanation = explainer.explain_instance(
            selected_case[0],
            model.predict_proba
        )
        st.subheader("LIME Explanation")
        explanation.show_in_notebook()  # For debugging; will show in notebook
        st.pyplot(explanation.as_pyplot_figure())  # Streamlit-compatible LIME output

        # Prepare LLM Prompt
        feature_contributions = explanation.as_list()
        features_for_llm = ', '.join([f"{feature}: {round(contribution, 2)}" for feature, contribution in feature_contributions])
        prediction_label = "Fraud" if selected_row["Prediction"] == 1 else "Not Fraud"
        prediction_prob = round(selected_row["Fraud Probability"] * 100, 2)

        prompt = (
            f"The machine learning model has classified this transaction as '{prediction_label}' with a fraud probability of {prediction_prob}%. "
            f"The factors influencing this decision are: {features_for_llm}. "
        )

        if selected_row["Prediction"] == 1:
            prompt += "Explain why this transaction might be classified as fraud based on these factors."
        else:
            prompt += "Explain why this transaction is considered not fraudulent based on these factors."

        # LLM Explanation
        st.subheader("LLM Explanation")
        try:
            llm_explanation = get_completion(prompt)
            st.text_area("LLM Explanation", llm_explanation, height=200)
        except Exception as e:
            st.error(f"Error with LLM explanation: {e}")
