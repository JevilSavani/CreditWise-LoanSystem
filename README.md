# 🏦 CreditWise Loan Approval System

An AI-powered loan approval prediction system built with Machine Learning and deployed with Streamlit.

## 📋 Features

- **Real-time Loan Prediction**: Get instant loan approval predictions
- **User-Friendly Interface**: Beautiful Streamlit web interface
- **Multiple Input Parameters**: Considers income, credit score, employment status, and more
- **Probability Breakdown**: Shows confidence level and probability of approval/rejection

## 🛠️ Tech Stack

- **Python**: Programming language
- **Scikit-learn**: Machine learning model
- **Pandas**: Data manipulation
- **Streamlit**: Web framework for deployment

## 📁 Project Structure

```
CreditWise-LoanSystem/
├── app.py                  # Streamlit application
├── pipeline.pkl           # Trained model pipeline
├── loan_approval_data.csv # Training dataset
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # This file
```

## 🚀 Deployment to Streamlit Cloud

### Step 1: Prepare Your GitHub Repository

1. **Push your code to GitHub**:
   ```bash
   # Initialize git (if not already done)
   git init
   git add .
   git commit -m "Add Streamlit app for loan prediction"
   
   # Add your GitHub repository
   git remote add origin https://github.com/JevilSavani/CreditWise-LoanSystem.git
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[Streamlit Cloud](https://share.streamlit.io)**
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in the details:
   - **Repository**: `JevilSavani/CreditWise-LoanSystem`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy"**

### Step 3: Access Your App

Once deployed, your app will be available at:
```
https://creditwise-loansystem.streamlit.app
```

## 💻 Local Development

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JevilSavani/CreditWise-LoanSystem.git
   cd CreditWise-LoanSystem
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # Linux/Mac
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and go to `http://localhost:8501`

## 📊 How It Works

The model uses the following features to predict loan approval:

| Feature | Description |
|---------|-------------|
| Applicant Income | Monthly income of the applicant |
| Coapplicant Income | Monthly income of co-applicant |
| Credit Score | Credit score (300-850) |
| Age | Applicant's age |
| Employment Status | Salaried/Self-employed/Unemployed |
| Loan Amount | Requested loan amount |
| Loan Term | Repayment period in months |
| DTI Ratio | Debt-to-income ratio |
| Savings | Total savings |
| Collateral Value | Value of collateral |
| Property Area | Urban/Semiurban/Rural |
| And more... | Various categorical features |

## 🔧 Customization

### Modify the Model

If you want to retrain the model:

1. Open `Loan.ipynb` in Jupyter
2. Make changes to the model
3. Save the new pipeline:
   ```python
   with open('pipeline.pkl', 'wb') as f:
       pickle.dump(model, f)
   ```

### Modify the UI

Edit `app.py` to customize:
- Add/remove input fields
- Change styling
- Add more visualizations

## 📝 License

This project is for educational purposes.

## 👤 Author

- **Jevil Savani** - [GitHub](https://github.com/JevilSavani)

## 🙏 Acknowledgments

- Built with Streamlit
- Machine learning model using Scikit-learn