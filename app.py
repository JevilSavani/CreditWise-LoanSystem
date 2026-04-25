"""
CreditWise Loan Approval Prediction App
Streamlit Application for Loan Detection System
"""

import streamlit as st
import pandas as pd
import pickle
import os

# Page Configuration
st.set_page_config(
    page_title="CreditWise - Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    """Load the trained model pipeline"""
    model_path = os.path.join(os.path.dirname(__file__), 'pipeline.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Load the data to get feature information
@st.cache_data
def load_data_info():
    """Load sample data to understand feature categories"""
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'loan_approval_data.csv'))
    return df

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🏦 CreditWise Loan Approval System")
    st.markdown("### AI-Powered Loan Approval Prediction")
    st.markdown("---")
    
    # Load model and data
    try:
        model = load_model()
        df = load_data_info()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Sidebar
    st.sidebar.header("📋 Application Details")
    st.sidebar.info("Fill in the details to check your loan approval status")
    
    # Get unique values for categorical features
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
    
    # Prediction button
    st.markdown("---")
    
    if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
        # Create input dataframe matching training data format
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
        
        try:
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
            st.write("Please check all input values and try again.")
    
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