from openai import AzureOpenAI
import os
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


# prompts


# create_prompt
def create_prompt(features, shap_values, prediction_label, prediction_prob):
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

def create_prompt(features, shap_values, prediction_label, prediction_prob):
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
    Mention top contributors for prediction first whether it is positive or negative.
    Seperately mention positive and negative contributors with bullet points. 
    Disclaimer:
    These explanations are generated based on SHAP values, which are used to interpret the model's predictions. While SHAP provides insights into feature importance, these explanations are approximations and should be used as guidance rather than definitive reasoning.
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

# def create_prompt(features, shap_values):
#     explanation = "\n".join([f"{feature}: {value:.2f}" for feature, value in zip(features, shap_values)])
    
#     prompt = f"""
#     The following are the key factors that influence the prediction for the claim:

#     1. The feature names and their respective contributions (SHAP values) are listed below:
#     {explanation}

#     2. The prediction is based on these factors:
#         - Positive SHAP values indicate that the feature contributes to increasing the likelihood of fraud.
#         - Negative SHAP values suggest that the feature is associated with a lower likelihood of fraud.
#         - A value near zero means the feature has little to no impact on the prediction.

#     In this case:
#         - If a feature has a large positive value, it is pushing the model to classify the claim as fraudulent.
#         - If a feature has a large negative value, it is pulling the model towards classifying the claim as non-fraudulent.

#     Please focus on the features with the highest positive and negative SHAP values to understand what the model considers most important when making the decision.
#     """
    
#     return prompt
