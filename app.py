# app.py
import streamlit as st
from main import semantic_pattern_adverserial_analysis
import json

st.title("Transaction Risk Analysis (JSON Input)")
st.write("Enter a new transaction as JSON:")

json_input = st.text_area(
    "Transaction JSON", 
    '{"amount": 1000, "type": "CASH_IN", "oldbalanceOrg": 5000, "newbalanceOrig": 6000}'
)

if st.button("Analyze Transaction"):
    if json_input.strip() == "":
        st.warning("Please enter transaction data in JSON format.")
    else:
        try:
            transaction_data = json.loads(json_input)  # Parse JSON
            result = semantic_pattern_adverserial_analysis(transaction_data)

            # Display results in organized expanders
            for key, value in result.items():
                with st.expander(key.replace("_", " ").title(), expanded=False):
                    st.write(value)

        except json.JSONDecodeError:
            st.error("Invalid JSON! Please check your input and try again.")