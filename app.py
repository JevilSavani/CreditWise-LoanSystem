"""
CreditWise Loan Approval Prediction App
Streamlit Application for Loan Detection System
"""

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CreditWise - Loan Approval Prediction",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the trained model pipeline safely."""
    possible_paths = [
        "pipeline.pkl",
        os.path.join(os.path.dirname(__file__), "pipeline.pkl"),
    ]

    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                    repair_sklearn_imputers(model)
                    return model
        except Exception as exc:
            st.error(f"Could not load {model_path}: {exc}")
            return None

    st.error("Could not find pipeline.pkl")
    return None


def repair_sklearn_imputers(model):
    """Add sklearn's newer imputer dtype attributes to older pickled models."""
    try:
        estimators = model.get_params(deep=True).values()
    except Exception:
        return

    for estimator in estimators:
        if estimator.__class__.__name__ != "SimpleImputer":
            continue

        statistics = getattr(estimator, "statistics_", None)
        fallback_dtype = getattr(statistics, "dtype", np.dtype("float64"))
        fit_dtype = getattr(estimator, "_fit_dtype", fallback_dtype)

        if not hasattr(estimator, "_fit_dtype"):
            estimator._fit_dtype = fit_dtype

        if not hasattr(estimator, "_fill_dtype"):
            estimator._fill_dtype = fit_dtype


@st.cache_data
def load_training_data():
    """Load sample data to confirm the deployed data file is available."""
    possible_paths = [
        "loan_approval_data.csv",
        os.path.join(os.path.dirname(__file__), "loan_approval_data.csv"),
    ]

    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                return pd.read_csv(data_path)
        except Exception:
            continue

    return None


def build_model_input(
    expected_cols,
    applicant_income,
    coapplicant_income,
    employment_status_select,
    age,
    marital_status_select,
    dependents,
    credit_score,
    existing_loans,
    dti_ratio,
    savings,
    collateral_value,
    loan_amount,
    loan_term,
    loan_purpose_select,
    property_area_select,
    education_level_select,
    gender_select,
    employer_category_select,
):
    """Create the exact numeric/dummy feature frame expected by the model."""
    row = {col: 0 for col in expected_cols}

    numeric_values = {
        "Applicant_Income": applicant_income,
        "Coapplicant_Income": coapplicant_income,
        "Age": age,
        "Dependents": dependents,
        "Credit_Score": credit_score,
        "Existing_Loans": existing_loans,
        "DTI_Ratio": dti_ratio,
        "Savings": savings,
        "Collateral_Value": collateral_value,
        "Loan_Amount": loan_amount,
        "Loan_Term": loan_term,
        "Education_Level": 0 if education_level_select == "Graduate" else 1,
    }

    dummy_values = {
        f"Employment_Status_{employment_status_select}": 1,
        f"Marital_Status_{marital_status_select}": 1,
        f"Property_Area_{property_area_select}": 1,
        f"Employer_Category_{employer_category_select}": 1,
        f"Gender_{gender_select}": 1,
        f"Loan_Purpose_{loan_purpose_select}": 1,
    }

    for col, value in {**numeric_values, **dummy_values}.items():
        if col in row:
            row[col] = value

    return pd.DataFrame([row], columns=expected_cols)


def main():
    """Main Streamlit application."""
    st.title("CreditWise Loan Approval System")
    st.markdown("### AI-Powered Loan Approval Prediction")
    st.markdown("---")

    st.sidebar.header("Application Details")
    st.sidebar.info("Fill in the details to check your loan approval status")

    employment_status = ["Salaried", "Self-employed", "Contract", "Unemployed"]
    marital_status = ["Married", "Single"]
    property_area = ["Urban", "Semiurban", "Rural"]
    employer_category = ["Private", "Government", "Unemployed", "MNC", "Business"]
    gender = ["Female", "Male"]
    loan_purpose = ["Personal", "Car", "Business", "Home", "Education"]
    education_level = ["Not Graduate", "Graduate"]

    st.markdown("### Applicant Information")
    col1, col2 = st.columns(2)

    with col1:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

    with col2:
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=2000, step=500)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0, step=1)

    st.markdown("### Loan Details")
    col3, col4 = st.columns(2)

    with col3:
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=25000, step=1000)
        loan_term = st.selectbox("Loan Term (months)", options=[12, 24, 36, 48, 60, 72, 84], index=4)
        dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.05)

    with col4:
        savings = st.number_input("Savings ($)", min_value=0, value=10000, step=1000)
        collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=30000, step=5000)
        property_area_select = st.selectbox("Property Area", options=property_area)

    st.markdown("### Additional Information")
    col5, col6 = st.columns(2)

    with col5:
        employment_status_select = st.selectbox("Employment Status", options=employment_status)
        marital_status_select = st.selectbox("Marital Status", options=marital_status)
        gender_select = st.selectbox("Gender", options=gender)

    with col6:
        employer_category_select = st.selectbox("Employer Category", options=employer_category)
        education_level_select = st.selectbox("Education Level", options=education_level)
        loan_purpose_select = st.selectbox("Loan Purpose", options=loan_purpose)

    st.markdown("---")

    if st.button(
        "Predict Loan Approval",
        type="primary",
        use_container_width=True,
        key="predict_loan_approval_button",
    ):
        try:
            model = load_model()

            if model is None:
                st.error("Failed to load model.")
                return

            if hasattr(model, "feature_names_in_"):
                expected_cols = model.feature_names_in_.tolist()
            else:
                st.error("Model does not have feature_names_in_. Please retrain.")
                return

            input_data = build_model_input(
                expected_cols,
                applicant_income,
                coapplicant_income,
                employment_status_select,
                age,
                marital_status_select,
                dependents,
                credit_score,
                existing_loans,
                dti_ratio,
                savings,
                collateral_value,
                loan_amount,
                loan_term,
                loan_purpose_select,
                property_area_select,
                education_level_select,
                gender_select,
                employer_category_select,
            )

            with st.expander("Debug: Input Data"):
                st.write("Input shape:", input_data.shape)
                st.write("Input columns:", input_data.columns.tolist())
                st.write("Input data:", input_data)

            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)

            st.markdown("### Prediction Result")

            if prediction[0] == 1:
                st.success("Congratulations! Your loan is likely to be APPROVED!")
                st.balloons()
            else:
                st.error("Sorry, your loan application is likely to be REJECTED.")

            confidence = prediction_proba[0][prediction[0]] * 100
            st.markdown(f"**Confidence Level:** {confidence:.2f}%")

            st.markdown("#### Probability Breakdown:")
            prob_df = pd.DataFrame(
                {
                    "Status": ["Rejected", "Approved"],
                    "Probability": [prediction_proba[0][0] * 100, prediction_proba[0][1] * 100],
                }
            )
            st.bar_chart(prob_df.set_index("Status"))

        except Exception as exc:
            st.error(f"Error making prediction: {exc}")
            st.write("---")
            st.write("Please check all input values and try again.")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>CreditWise Loan Detection System | Powered by Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
