import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ────────────────────────────────────────────────
# Page config – changed to wide layout
# ────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Insurance Charge Predictor",
    page_icon="💰",
    layout="wide"          # Changed from "centered" to "wide"
)

# ────────────────────────────────────────────────
# Load model
# ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("best_health_insurance_model.pkl")

model = load_model()

EXPECTED_COLS = ["age", "sex", "bmi", "children", "smoker", "region"]

# ────────────────────────────────────────────────
# Title
# ────────────────────────────────────────────────
st.title("Health Insurance Charge Predictor")
st.markdown(
    "Predict **annual health insurance charges** based on personal and lifestyle factors. "
    "Get insights into your potential insurance costs."
)

# =================================================
# 1. SINGLE PREDICTION
# =================================================
st.header("Individual Prediction")

with st.form("single_prediction"):
    c1, c2 = st.columns(2)

    with c1:
        age = st.number_input("Age", 18, 100, 30)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, step=0.1)
        children = st.number_input("Number of children", 0, 10, 0)

    with c2:
        sex = st.selectbox("Sex", ["female", "male"])
        smoker = st.selectbox("Smoker?", ["no", "yes"])
        region = st.selectbox(
            "Region", ["northeast", "northwest", "southeast", "southwest"]
        )

    submit_single = st.form_submit_button("Calculate Estimated Charge", type="primary")

if submit_single:
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    pred_log = model.predict(input_df)[0]
    pred = np.expm1(pred_log)

    st.success("Prediction complete")
    st.metric("Estimated Annual Charge", f"${pred:,.2f}")

# =================================================
# 2. BATCH PROCESSING
# =================================================
st.header("Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    missing = set(EXPECTED_COLS) - set(batch_df.columns)
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    preds_log = model.predict(batch_df[EXPECTED_COLS])
    batch_df["predicted_charge"] = np.expm1(preds_log)

    st.session_state["batch_results"] = batch_df
    st.success("Batch prediction completed")
    st.dataframe(batch_df)

# =================================================
# 3. POST-BATCH ANALYSIS
# =================================================
if "batch_results" in st.session_state:
    st.header("Batch Analysis")

    df = st.session_state["batch_results"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Charge", f"${df.predicted_charge.mean():,.0f}")
    c2.metric("Maximum Charge", f"${df.predicted_charge.max():,.0f}")
    c3.metric("High-Risk Count", (df.predicted_charge > 20000).sum())

    def risk_bucket(x):
        if x < 5000:
            return "Low"
        elif x < 15000:
            return "Medium"
        return "High"

    df["risk_category"] = df.predicted_charge.apply(risk_bucket)
    st.subheader("Risk Distribution")
    st.bar_chart(df["risk_category"].value_counts())

# =================================================
# 5. AI ASSISTANT (BATCH INTERACTION)
# =================================================
if "batch_results" in st.session_state:
    st.header("AI Assistant – Batch Insights")

    question = st.text_area(
        "Ask about the batch results (e.g. 'How many high risk smokers?')"
    )

    df = st.session_state["batch_results"]

    if question:
        q = question.lower()

        if "high" in q and "risk" in q:
            count = (df.predicted_charge > 20000).sum()
            st.write(f"There are **{count} high-risk individuals**.")

        elif "smoker" in q:
            avg = df[df.smoker == "yes"].predicted_charge.mean()
            st.write(f"Average predicted charge for smokers: **${avg:,.0f}**")

        elif "average" in q:
            st.write(f"Overall average charge: **${df.predicted_charge.mean():,.0f}**")

        else:
            st.write(
                "I can answer questions about risk levels, smokers, or average costs."
            )

# ────────────────────────────────────────────────
# Footer
# ────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Developed By: Lesson Karidza | Data Scientist| [GitHub](https://github.com/Lesson2003) | [LinkedIn](https://www.linkedin.com/in/lessonshepherdkaridza/) | [Portfolio](https://www.datascienceportfolio.io/lessonshepherdkaridz.com)"
)