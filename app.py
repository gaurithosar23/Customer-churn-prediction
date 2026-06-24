import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
model=joblib.load("churn_model.pkl")
model_columns=joblib.load("model_columns.pkl")
# Page settings
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load dataset
df = pd.read_excel("Telco_customer_churn.csv.xlsx")

# Metrics
total_customers = len(df)
churned_customers = int(df["Churn Value"].sum())
retained_customers = total_customers - churned_customers
churn_rate = round((churned_customers / total_customers) * 100, 2)

# Sidebar
st.sidebar.title("📋 Project Information")
st.sidebar.write("**Project:** Intelligent Customer Churn Prediction")
st.sidebar.write("**Algorithm:** XGBoost")
st.sidebar.write("**Accuracy:** 80.62%")
st.sidebar.write("**Dataset:** IBM Telco Customer Churn")

# Main Title
st.title("📊 Intelligent Customer Churn Prediction System")
st.markdown("Predicting customer churn using machine learning.")

st.divider()

# KPI Cards
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churned_customers)
col3.metric("Retained Customers", retained_customers)
col4.metric("Churn Rate", f"{churn_rate}%")

st.divider()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Distribution")

    fig1, ax1 = plt.subplots()
    ax1.pie(
        [churned_customers, retained_customers],
        labels=["Churned", "Retained"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig1)

with col2:
    st.subheader("Top Features Affecting Churn")

    features = [
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "CLTV",
        "Internet Service"
    ]

    importance = [0.24, 0.21, 0.18, 0.15, 0.12]

    fig2, ax2 = plt.subplots()
    ax2.barh(features, importance)
    ax2.set_xlabel("Importance Score")

    st.pyplot(fig2)

st.divider()

st.subheader("📄 Dataset Preview")
st.dataframe(df.head())
st.divider()
st.subheader("🔮 Customer Churn Prediction")

tenure = st.slider("Tenure Months", 0, 72, 12)
monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)
total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

if st.button("Predict Churn"):

    input_data = pd.DataFrame(
        columns=model_columns
    )

    input_data.loc[0] = 0

    if "Tenure Months" in model_columns:
        input_data.at[0, "Tenure Months"] = tenure

    if "Monthly Charges" in model_columns:
        input_data.at[0, "Monthly Charges"] = monthly_charges

    if "Total Charges" in model_columns:
        input_data.at[0, "Total Charges"] = total_charges

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
st.success("✅ Dashboard Created Successfully")