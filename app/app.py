import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import streamlit as st
from predict import predict_churn

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="centered",
)

st.markdown(
    """
    <style>
        .stApp { background: #0b0f19; color: #f5f7fb; }
        .app-header { text-align: center; margin-bottom: 1.2rem; }
        .app-header h1 { font-size: 1.8rem; margin-bottom: 0.2rem; }
        .app-header p { color: #94a3b8; font-size: 0.9rem; }
        .risk-box {
            padding: 1.2rem; border-radius: 14px; text-align: center;
            margin-top: 1rem;
        }
        .risk-high { background: rgba(248,113,113,0.15); border: 1px solid rgba(248,113,113,0.4); }
        .risk-low { background: rgba(52,211,153,0.15); border: 1px solid rgba(52,211,153,0.4); }
        .risk-box h2 { margin: 0 0 0.3rem 0; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1>📉 Customer Churn Predictor</h1>
        <p>Enter a customer's profile to estimate their likelihood of churning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("churn_form"):
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=0.5)
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)), step=1.0
        )

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        result = predict_churn(customer)

        risk_class = "risk-high" if result["will_churn"] else "risk-low"
        icon = "⚠️" if result["will_churn"] else "✅"

        st.markdown(
            f"""
            <div class="risk-box {risk_class}">
                <h2>{icon} {result['prediction']}</h2>
                <p>Estimated churn probability: <b>{result['churn_probability'] * 100:.1f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.progress(result["churn_probability"])

        with st.expander("What influences this?"):
            st.markdown(
                "- **Short tenure + month-to-month contract** is historically the "
                "strongest churn signal in this dataset.\n"
                "- **No security/tech-support add-ons** correlates with higher churn.\n"
                "- **Fiber optic internet with high monthly charges** and "
                "**electronic check payment** are also associated with higher churn.\n\n"
                "See `reports/shap_summary.png` (generated by `src/train_model.py`) "
                "for the model's actual learned feature importances."
            )

    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python src/train_model.py` first "
            "to train and save the model."
        )
