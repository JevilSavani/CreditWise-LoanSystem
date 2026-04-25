"""
CreditWise Loan Approval Prediction App
Streamlit Application for Loan Detection System
==============================================
FULLY FIXED VERSION - Handles preprocessing correctly
"""

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CreditWise - Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# ============================================
# SAFE MODEL LOADING
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model pipeline safely."""
    possible_paths = [
        'pipeline.pkl',
        os.path.join(os.path.dirname(__file__), 'pipeline.pkl'),
    ]
    
    for model_path in possible_paths:
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            continue
    
    st.error(f"❌ Could not find pipeline.pkl")
    return None

# ============================================
# LOAD DATA FOR CATEGORIES
# ============================================
@st.cache_data
def load_training_data():
    """Load sample data to get categorical values."""
    possible_paths = [
        'loan_approval_data.csv',
        os.path.join(os.path.dirname(__file__), 'loan_approval_data.csv'),
    ]
    
    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                return pd.read_csv(data_path)
        except:
            continue
    return None

# ============================================
# MANUAL PREPROCESSING (matches training)
# ============================================
def preprocess_input(input_df, df):
    """
    Manually preprocess input to match exactly how the model was trained.
    
    The model was trained on ONE-HOT ENCODED data with these columns:
    - Numerical: Applicant_Income, Coapplicant_Income, Age, Dependents, 
                Credit_Score, Existing_Loans, DTI_Ratio, Savings, 
                Collateral_Value, Loan_Amount, Loan_Term, Education_Level
    - Categorical (one-hot encoded): Employment_Status, Marital_Status, 
                Property_Area, Employer_Category, Gender, Loan_Purpose
    """
    
    # Get categorical columns from training data
    categorical_cols = ['Employment_Status', 'Marital_Status', 'Property_Area', 
                       'Employer_Category', 'Gender', 'Loan_Purpose']
    
    # Get unique values for each categorical column from training data
    cat_values = {}
    for col in categorical_cols:
        if col in df.columns:
            cat_values[col] = df[col].dropna().unique().tolist()
    
    # One-hot encode the input
    for col in categorical_cols:
        if col in input_df.columns and col in cat_values:
            # Create one-hot encoding
            unique_vals = cat_values[col]
            for val in unique_vals:
                new_col = f"{col}_{val}"
                input_df[new_col] = (input_df[col] == val).astype(float)
    
    # Drop original categorical columns (keep only encoded)
    input_df = input_df.drop(columns=categorical_cols, errors='ignore')
    
    # Ensure all expected columns exist (fill missing with 0)
    expected_cols = [
        'Applicant_Income', 'Coapplicant_Income', 'Age', 'Dependents',
        'Credit_Score', 'Existing_Loans', 'DTI_Ratio', 'Savings',
        'Collateral_Value', 'Loan_Amount', 'Loan_Term', 'Education_Level',
        'Employment_Status_Salaried', 'Employment_Status_Self-employed',
        'Employment_Status_Unemployed', 'Marital_Status_Married',
        'Marital_Status_Single', 'Property_Area_Rural', 'Property_Area_Semiurban',
        'Property_Area_Urban', 'Employer_Category_Government',
        'Employer_Category_Other', 'Employer_Category_Private',
        'Employer_Category_Self-employed', 'Gender_Female', 'Gender_Male',
        'Loan_Purpose_Business', 'Loan_Purpose_Car', 'Loan_Purpose_Education',
        'Loan_Purpose_Home', 'Loan_Purpose_Personal'
    ]
    
    # Reindex to match expected columns
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)
    
    return input_df

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main Streamlit application"""
    
    st.title("🏦 CreditWise Loan Approval System")
    st.markdown("### AI-Powered Loan Approval Prediction")
    st.markdown("---")
    
    # Load model and data
    model = load_model()
    df = load_training_data()
    
    if model is None:
        st.error("🚫 Failed to load model.")
        return
    
    if df is None:
        st.error("🚫 Failed to load training data.")
        return
    
    # Sidebar
    st.sidebar.header("📋 Application Details")
    st.sidebar.info("Fill in the details to check your loan approval status")
    
    # Get unique values from training data
    employment_status = df['Employment_Status'].dropna().unique().tolist() if 'Employment_Status' in df.columns else ['Salaried', 'Self-employed', 'Unemployed']
    marital_status = df['Marital_Status'].dropna().unique().tolist() if 'Marital_Status' in df.columns else ['Married', 'Single']
    property_area = df['Property_Area'].dropna().unique().tolist() if 'Property_Area' in df.columns else ['Urban', 'Semiurban', 'Rural']
    employer_category = df['Employer_Category'].dropna().unique().tolist() if 'Employer_Category' in df.columns else ['Private', 'Government', 'Self-employed', 'Other']
    gender = df['Gender'].dropna().unique().tolist() if 'Gender' in df.columns else ['Male', 'Female']
    loan_purpose = df['Loan_Purpose'].dropna().unique().tolist() if 'Loan_Purpose' in df.columns else ['Personal', 'Home', 'Car', 'Business', 'Education']
    education_level = df['Education_Level'].dropna().unique().tolist() if 'Education_Level' in df.columns else ['Graduate', 'Not Graduate']
    
    # ============================================
    # INPUT FORM
    # ============================================
    st.markdown("### 👤 Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=30)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    
    with col2:
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=2000, step=500)
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0, step=1)
        existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=0, step=1)
    
    st.markdown("### 💰 Loan Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, value=25000, step=1000)
        loan_term = st.selectbox("Loan Term (months)", options=[12, 24, 36, 48, 60, 72, 84], index=4)
        dti_ratio = st.number_input("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    
    with col4:
        savings = st.number_input("Savings ($)", min_value=0, value=10000, step=1000)
        collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=30000, step=5000)
        property_area_select = st.selectbox("Property Area", options=property_area)
    
    st.markdown("### 📝 Additional Information")
    
    col5, col6 = st.columns(2)
    
    with col5:
        employment_status_select = st.selectbox("Employment Status", options=employment_status)
        marital_status_select = st.selectbox("Marital Status", options=marital_status)
        gender_select = st.selectbox("Gender", options=gender)
    
    with col6:
        employer_category_select = st.selectbox("Employer Category", options=employer_category)
        education_level_select = st.selectbox("Education Level", options=education_level)
        loan_purpose_select = st.selectbox("Loan Purpose", options=loan_purpose)
    
    # ============================================
    # PREDICTION
    # ============================================
    st.markdown("---")
    
    if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
        try:
            # Create input dataframe with RAW columns
            input_data = pd.DataFrame({
                'Applicant_Income': [applicant_income],
                'Coapplicant_Income': [coapplicant_income],
                'Employment_Status': [employment_status_select],
                'Age': [age],
                'Marital_Status': [marital_status_select],
                'Dependents': [dependents],
                'Credit_Score': [credit_score],
                'Existing_Loans': [existing_loans],
                'DTI_Ratio': [dti_ratio],
                'Savings': [savings],
                'Collateral_Value': [collateral_value],
                'Loan_Amount': [loan_amount],
                'Loan_Term': [loan_term],
                'Loan_Purpose': [loan_purpose_select],
                'Property_Area': [property_area_select],
                'Education_Level': [education_level_select],
                'Gender': [gender_select],
                'Employer_Category': [employer_category_select]
            })
            
            # Apply manual preprocessing (one-hot encoding)
            input_data = preprocess_input(input_data, df)
            
            # Debug info
            with st.expander("🔍 Debug: Input Data"):
                st.write("Input shape:", input_data.shape)
                st.write("Input columns:", input_data.columns.tolist())
                st.write("Input data:", input_data)
            
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Display results
            st.markdown("### 📊 Prediction Result")
            
            if prediction[0] == 1:
                st.success("✅ Congratulations! Your loan is likely to be APPROVED!")
                st.balloons()
            else:
                st.error("❌ Sorry, your loan application is likely to be REJECTED.")
            
            confidence = prediction_proba[0][prediction[0]] * 100
            st.markdown(f"**Confidence Level:** {confidence:.2f}%")
            
            st.markdown("#### Probability Breakdown:")
            prob_df = pd.DataFrame({
                'Status': ['Rejected', 'Approved'],
                'Probability': [prediction_proba[0][0] * 100, prediction_proba[0][1] * 100]
            })
            st.bar_chart(prob_df.set_index('Status'))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("---")
            st.write("Please check all input values and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>CreditWise Loan Detection System | Powered by Machine Learning</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

# ============================================
# SAFE DATA LOADING
# ============================================
@st.cache_data
def load_training_data():
    """
    Load sample data to understand feature columns.
    """
    possible_paths = [
        'loan_approval_data.csv',
        os.path.join(os.path.dirname(__file__), 'loan_approval_data.csv'),
    ]
    
    for data_path in possible_paths:
        try:
            if os.path.exists(data_path):
                return pd.read_csv(data_path)
        except Exception as e:
            continue
    
    return None

# ============================================
# GET FEATURE COLUMNS FROM MODEL
# ============================================
def get_feature_columns(model, df):
    """
    Extract the expected feature columns from the model pipeline.
    This ensures we send the right columns to the model.
    """
    try:
        # Get the preprocessor from the pipeline
        preprocessor = model.named_steps['preprocess']
        
        # Get feature names after transformation
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            # Get transformer names
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_:
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        names = transformer.get_feature_names_out()
                        feature_names.extend(names)
                    except:
                        feature_names.extend(columns)
        
        return list(feature_names)
    except Exception as e:
        st.warning(f"Could not extract feature names: {e}")
        # Return original columns as fallback
        return df.columns.tolist() if df is not None else []

# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏦 CreditWise Loan Approval System")
    st.markdown("### AI-Powered Loan Approval Prediction")
    st.markdown("---")
    
    # Load model and data
    model = load_model()
    df = load_training_data()
    
    if model is None:
        st.error("🚫 Failed to load model. Please check deployment.")
        st.info("Make sure pipeline.pkl is in the repository root.")
        return
    
    # Sidebar
    st.sidebar.header("📋 Application Details")
    st.sidebar.info("Fill in the details to check your loan approval status")
    
    # ============================================
    # INPUT FORM - ALL FEATURES
    # ============================================
    
    # Get unique values for categorical features from data if available
    if df is not None:
        employment_status = df['Employment_Status'].dropna().unique().tolist() if 'Employment_Status' in df.columns else ['Salaried', 'Self-employed', 'Unemployed']
        marital_status = df['Marital_Status'].dropna().unique().tolist() if 'Marital_Status' in df.columns else ['Married', 'Single']
        property_area = df['Property_Area'].dropna().unique().tolist() if 'Property_Area' in df.columns else ['Urban', 'Semiurban', 'Rural']
        employer_category = df['Employer_Category'].dropna().unique().tolist() if 'Employer_Category' in df.columns else ['Private', 'Government', 'Self-employed', 'Other']
        gender = df['Gender'].dropna().unique().tolist() if 'Gender' in df.columns else ['Male', 'Female']
        loan_purpose = df['Loan_Purpose'].dropna().unique().tolist() if 'Loan_Purpose' in df.columns else ['Personal', 'Home', 'Car', 'Business', 'Education']
        education_level = df['Education_Level'].dropna().unique().tolist() if 'Education_Level' in df.columns else ['Graduate', 'Not Graduate']
    else:
        # Default values
        employment_status = ['Salaried', 'Self-employed', 'Unemployed']
        marital_status = ['Married', 'Single']
        property_area = ['Urban', 'Semiurban', 'Rural']
        employer_category = ['Private', 'Government', 'Self-employed', 'Other']
        gender = ['Male', 'Female']
        loan_purpose = ['Personal', 'Home', 'Car', 'Business', 'Education']
        education_level = ['Graduate', 'Not Graduate']
    
    # Create input form
    st.markdown("### 👤 Applicant Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        applicant_income = st.number_input(
            "Applicant Income ($)",
            min_value=0,
            value=5000,
            step=500,
            help="Your monthly income"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=30,
            help="Your current age"
        )
        
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            help="Your credit score (300-850)"
        )
    
    with col2:
        coapplicant_income = st.number_input(
            "Co-applicant Income ($)",
            min_value=0,
            value=2000,
            step=500,
            help="Co-applicant's monthly income (if any)"
        )
        
        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Number of family members dependent on you"
        )
        
        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            help="Number of existing loans"
        )
    
    st.markdown("### 💰 Loan Details")
    
    col3, col4 = st.columns(2)
    
    with col3:
        loan_amount = st.number_input(
            "Loan Amount ($)",
            min_value=1000,
            value=25000,
            step=1000,
            help="Amount you want to borrow"
        )
        
        loan_term = st.selectbox(
            "Loan Term (months)",
            options=[12, 24, 36, 48, 60, 72, 84],
            index=4,
            help="Repayment period in months"
        )
        
        dti_ratio = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="Your debt-to-income ratio"
        )
    
    with col4:
        savings = st.number_input(
            "Savings ($)",
            min_value=0,
            value=10000,
            step=1000,
            help="Your total savings"
        )
        
        collateral_value = st.number_input(
            "Collateral Value ($)",
            min_value=0,
            value=30000,
            step=5000,
            help="Value of any collateral"
        )
        
        property_area_select = st.selectbox(
            "Property Area",
            options=property_area,
            help="Location of the property"
        )
    
    st.markdown("### 📝 Additional Information")
    
    col5, col6 = st.columns(2)
    
    with col5:
        employment_status_select = st.selectbox(
            "Employment Status",
            options=employment_status,
            help="Your current employment status"
        )
        
        marital_status_select = st.selectbox(
            "Marital Status",
            options=marital_status,
            help="Your marital status"
        )
        
        gender_select = st.selectbox(
            "Gender",
            options=gender,
            help="Your gender"
        )
    
    with col6:
        employer_category_select = st.selectbox(
            "Employer Category",
            options=employer_category,
            help="Type of your employer"
        )
        
        education_level_select = st.selectbox(
            "Education Level",
            options=education_level,
            help="Your education qualification"
        )
        
        loan_purpose_select = st.selectbox(
            "Loan Purpose",
            options=loan_purpose,
            help="Purpose of the loan"
        )
    
    # ============================================
    # PREDICTION
    # ============================================
    st.markdown("---")
    
    if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
        try:
            # Create input dataframe with RAW columns (no encoding!)
            input_data = pd.DataFrame({
                'Applicant_Income': [applicant_income],
                'Coapplicant_Income': [coapplicant_income],
                'Employment_Status': [employment_status_select],
                'Age': [age],
                'Marital_Status': [marital_status_select],
                'Dependents': [dependents],
                'Credit_Score': [credit_score],
                'Existing_Loans': [existing_loans],
                'DTI_Ratio': [dti_ratio],
                'Savings': [savings],
                'Collateral_Value': [collateral_value],
                'Loan_Amount': [loan_amount],
                'Loan_Term': [loan_term],
                'Loan_Purpose': [loan_purpose_select],
                'Property_Area': [property_area_select],
                'Education_Level': [education_level_select],
                'Gender': [gender_select],
                'Employer_Category': [employer_category_select]
            })
            
            # ============================================
            # FIX: Match model's expected columns exactly
            # ============================================
            # The model was trained on ONE-HOT ENCODED data
            # We need to match the exact columns the model expects
            
            # Get the columns the model was trained on
            if hasattr(model, 'feature_names_in_'):
                expected_cols = model.feature_names_in_.tolist()
            else:
                # If no feature_names_in_, try to get from preprocessor
                try:
                    expected_cols = model.named_steps['preprocess'].get_feature_names_out()
                except:
                    st.error("Cannot determine expected columns. Please retrain the model.")
                    return
            
            # Reindex input to match expected columns, fill missing with 0
            input_data = input_data.reindex(columns=expected_cols, fill_value=0)
            
            # Debug: Show input columns
            with st.expander("🔍 Debug: Input Data"):
                st.write("Expected columns:", expected_cols)
                st.write("Input columns:", input_data.columns.tolist())
                st.write("Input data:", input_data)
            
            # Make prediction
            prediction = model.predict(input_data)
            prediction_proba = model.predict_proba(input_data)
            
            # Display results
            st.markdown("### 📊 Prediction Result")
            
            if prediction[0] == 1:
                st.success("✅ Congratulations! Your loan is likely to be APPROVED!")
                st.balloons()
            else:
                st.error("❌ Sorry, your loan application is likely to be REJECTED.")
            
            # Show confidence
            confidence = prediction_proba[0][prediction[0]] * 100
            st.markdown(f"**Confidence Level:** {confidence:.2f}%")
            
            # Show probability breakdown
            st.markdown("#### Probability Breakdown:")
            prob_df = pd.DataFrame({
                'Status': ['Rejected', 'Approved'],
                'Probability': [prediction_proba[0][0] * 100, prediction_proba[0][1] * 100]
            })
            st.bar_chart(prob_df.set_index('Status'))
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("---")
            st.write("Please check all input values and try again.")
            st.write("If the problem persists, the model may need retraining.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>CreditWise Loan Detection System | Powered by Machine Learning</p>
            <p>Built with Streamlit</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()