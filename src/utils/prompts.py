from langchain_core.prompts import PromptTemplate

semantic_grader_prompt = \
PromptTemplate.from_template(template=""" You are a financial risk assessment AI tasked with evaluating the risk of a transaction

You are given the following features of this individual transactions
{context}

Your task is to:

1. Analyze the transaction and output a **risk score** between 0 (no risk) and 1 (high risk).
2. Explain **why the model assigned this risk score**, referencing the input features and their values. Be detailed and specific.
3. Recommend whether the transaction **should be allowed, flagged for review, or require human intervention such as 2FA**, based on the risk assessment.
4. Provide clear reasoning that can be reviewed by a human compliance officer.

Here is an example input transaction (replace with actual values):

Risk Score: a float value between 0 and 1
Analysis: Gives the explanation on why the risk score was assigned, referencing specific features and their values.

Return in the following JSON format:
{{
    "risk_score": "<a float value between 0 and 1>",
    "analysis": "<Explain the risk factors, feature contributions, and unusual patterns in the transaction>",
}}
""")


pattern_grader_prompt = \
PromptTemplate.from_template(
"""You are an ensemble agent that combines predictions from multiple models
to assess the risk of a financial transaction being fraudulent.
The given input:

{user_input}

Here are the model metrics
    "xgboost": {{"AUC": 0.8757, "F1": 0.8097, "Precision": 0.7335, "Recall": 0.9034}},
    "lightgbm": {{"AUC": 0.8504, "F1": 0.7907, "Precision": 0.7129, "Recall": 0.8876}},
    "tabnet": {{"AUC": 0.9115, "F1": 0.8425, "Precision": 0.7829, "Recall": 0.9120}},

Given the the following averaged risk assessments from each model:
{risk_score}
Use the ensemble model to generate a final risk score and analysis.
Return in the following JSON format:
{{
    "risk_score": "<a float value between 0 and 1>",
    "analysis": "<Explain the risk factors, feature contributions, and unusual patterns in the transaction>",
}}
""")


decider_grader_prompt = \
PromptTemplate.from_template(
"""You are a decision-making AI that combines risk assessments from multiple experts
to determine the final risk score for a financial transaction.
Here are te given risk assessment values from different experts

Pattern Recognition Expert: {pattern_risk_score}
Semantic Analysis Expert: {semantic_risk_score}

Following this, here are the given detailed report from each expert:
Pattern Recognition Expert Report: {pattern_analysis}
Semantic Analysis Expert Report: {semantic_analysis}

After careful consideration of the provided risk assessments and reports, please give the final risk score for the transaction 
and a detailed report regarding why this is considered to be the final risk score.

Use the assessments to generate a final risk score and analysis.make sure to highlight the key factors from each expert's report that influenced your decision.
Return in the following JSON format:
{{
    "final_risk_score": "<a float value between 0 and 1>",
    "final_analysis": "<Explain the combined risk factors, feature contributions, and unusual patterns in the
    transaction>",
}}
""")

prompts = {
    "semantic_grader": semantic_grader_prompt,
    "pattern_grader": pattern_grader_prompt,
    "decider_grader": decider_grader_prompt
}