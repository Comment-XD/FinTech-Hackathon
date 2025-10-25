import time
from collections import deque

import streamlit as st
import pandas as pd
import numpy as np
from main import semantic_pattern_adverserial_analysis

st.set_page_config(page_title="FinanceAdvisor — Fraud Traffic Simulator", layout="wide")

st.title("FinanceAdvisor — Fraud Traffic Simulator")
st.markdown("Upload a transaction CSV or simulate streaming transactions. Each transaction is assessed and displayed with a semantic, pattern, and final score.")

# UI: upload or synthetic
with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
    use_sample = st.checkbox("Use built-in sample (synthetic)", value=not bool(uploaded))
    speed = st.slider("Stream speed (seconds per transaction)", min_value=0.1, max_value=3.0, value=0.8, step=0.1)
    history_k = st.number_input("History K (for DB lookup)", min_value=0, max_value=100, value=10, step=1)
    max_rows = st.number_input("Max rows to simulate", min_value=1, max_value=10000, value=200, step=1)
    run_button = st.button("Start / Restart simulation")
    stop_button = st.button("Stop")

# load DataFrame
def load_transactions():
    if uploaded:
        try:
            return pd.read_csv(uploaded)
        except Exception:
            return pd.DataFrame()
    if use_sample:
        # synthetic sample: columns similar to credit card / transaction dataset
        rng = np.random.default_rng(123)
        n = 1000
        df = pd.DataFrame({
            "step": np.arange(n),
            "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"], size=n, p=[0.4,0.3,0.2,0.1]),
            "amount": np.round(np.exp(rng.normal(7, 1, size=n))),  # skewed amounts
            "nameOrig": rng.choice([f"C{str(i).zfill(9)}" for i in range(1000, 1020)], size=n),
            "oldbalanceOrg": np.round(rng.uniform(0, 20000, size=n),2),
            "newbalanceOrig": np.round(rng.uniform(0, 200000, size=n),2),
            "nameDest": rng.choice([f"C{str(i).zfill(9)}" for i in range(2000, 2020)], size=n),
            "oldbalanceDest": np.round(rng.uniform(0, 500000, size=n),2),
            "newbalanceDest": np.round(rng.uniform(0, 500000, size=n),2),
            "isFraud": rng.choice([0,1], size=n, p=[0.995,0.005])
        })
        return df
    return pd.DataFrame()

df = load_transactions()
if df.empty:
    st.warning("No transactions loaded. Upload CSV or enable sample.")
    st.stop()

# session state
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "index" not in st.session_state:
    st.session_state.index = 0
if "scores" not in st.session_state:
    st.session_state.scores = deque(maxlen=500)
if "flagged" not in st.session_state:
    st.session_state.flagged = []

if run_button:
    st.session_state.sim_running = True
    st.session_state.index = 0
    st.session_state.scores.clear()
    st.session_state.flagged = []

if stop_button:
    st.session_state.sim_running = False

# layout panels
left, right = st.columns([2, 1])

with left:
    st.subheader("Live transaction stream")
    stream_box = st.empty()
    details_box = st.empty()

with right:
    st.subheader("Dashboard")
    score_chart = st.line_chart(pd.DataFrame({"final": []}))
    hist_chart = st.bar_chart(pd.DataFrame({"count": []}))
    st.markdown("### Top flagged transactions")
    flagged_table = st.empty()
    st.markdown("### Metrics")
    metrics_area = st.empty()


stop = False
if st.session_state.sim_running:
    rows = df.to_dict(orient="records")
    n = min(len(rows), int(max_rows))
    for i in range(st.session_state.index, n):
        if not st.session_state.sim_running:
            break
        row = rows[i]
        result = semantic_pattern_adverserial_analysis(row)
        
        semantic_score = result["semantic_risk_score"]
        pattern_score = result["pattern_risk_score"]
        final_score = result["final_risk_score"]
        semantic_analysis = result["semantic_analysis"]
        pattern_analysis = result["pattern_analysis"]
        history = "Simulated history data..." if history_k > 0 else "No history requested."
        
        # update session state
        st.session_state.index = i + 1
        st.session_state.scores.append(final_score)
        if final_score >= 0.7:
            st.session_state.flagged.append({**row, "final_score": final_score})

        # render stream item
        with stream_box.container():
            st.markdown(f"### Transaction #{i+1}")
            colA, colB = st.columns([2, 3])
            with colA:
                st.write("Transaction summary")
                st.json(row)
                st.write("History (excerpt):")
                if isinstance(history, str):
                    st.write(history if len(history) < 1000 else history[:1000] + "...")
                else:
                    st.write(history)
            with colB:
                st.metric("Semantic risk", f"{semantic_score:.3f}")
                st.metric("Pattern risk", f"{pattern_score:.3f}")
                st.metric("Final risk", f"{final_score:.3f}")
                st.markdown("**Explanation (semantic)**")
                st.write(semantic_analysis or "(no semantic explanation available)")
                st.markdown("**Pattern analysis (summary)**")
                st.write(f"Pattern risk {pattern_score:.3f} — see flagged table for details.")

        # update dashboard charts
        score_df = pd.DataFrame({"final": list(st.session_state.scores)})
        score_chart.add_rows(score_df.tail(1))
        # histogram of final scores
        hist_vals, edges = np.histogram(list(st.session_state.scores), bins=10, range=(0,1))
        hist_df = pd.DataFrame({"count": hist_vals}, index=[f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(edges)-1)])
        hist_chart.data = hist_df

        # flagged table
        if st.session_state.flagged:
            flagged_df = pd.DataFrame(st.session_state.flagged).sort_values("final_score", ascending=False).head(10)
            flagged_table.dataframe(flagged_df)
        else:
            flagged_table.write("No flagged transactions yet.")

        # metrics area
        metrics_area.markdown(f"- Processed: **{st.session_state.index}**\n- Avg final risk: **{np.mean(list(st.session_state.scores)):.3f}**\n- Flagged count: **{len(st.session_state.flagged)}**")

        time.sleep(speed)
        # allow stop button to take effect
        if st.session_state.index >= n:
            st.session_state.sim_running = False
            break

# show example output nicely
st.markdown("---")
st.subheader("Example combined assessment")
st.write("This is a formatted example output (from a previous run).")
st.json(EXAMPLE_OUTPUT)

st.markdown("Notes: This is a simulator. To use real models/LLM, ensure models and environment variables are configured and restart the app.")