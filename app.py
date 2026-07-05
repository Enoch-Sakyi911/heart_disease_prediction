import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import io

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"  # This ensures sidebar is OPEN by default
)

# -------------------------------------------------
# 2. LOAD THE SAVED MODEL AND SCALER
# -------------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('models/heart_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Feature names
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak']

# -------------------------------------------------
# 3. SIDEBAR - Professional Branding
# -------------------------------------------------
with st.sidebar:
    # Use a local emoji instead of external image URL (faster, no loading issues)
    st.markdown("## ❤️ Heart Disease")
    st.markdown("### Risk Predictor")

    st.markdown("---")
    st.markdown("### 🧠 Model Details")
    st.markdown("""
    - **Algorithm:** Logistic Regression  
    - **Accuracy:** 84.46%  
    - **AUC Score:** ~0.90  
    """)

    st.markdown("---")
    st.markdown("### 📋 Top Risk Factors")
    st.markdown("""
    1. Chest Pain Type (`cp`)  
    2. ST Depression (`oldpeak`)  
    3. Gender (Male)  
    4. Exercise Angina (`exang`)  
    5. Max Heart Rate (`thalach`)  
    """)
    
    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit | UCI Heart Disease Dataset")

# -------------------------------------------------
# 4. MAIN DASHBOARD - Tabs
# -------------------------------------------------
st.title("🫀 Heart Disease Prediction Dashboard")
st.caption("Upload a CSV file for batch predictions, or manually enter patient data below.")

# Create Tabs
tab1, tab2 = st.tabs([" Single Prediction", "📁 Batch CSV Upload"])

# -------------------------------------------------
# TAB 1: SINGLE PREDICTION
# -------------------------------------------------
with tab1:
    st.header("Enter Patient Data")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        age = st.number_input(" Age", min_value=20, max_value=100, value=50, step=1)
        sex = st.selectbox("⚤ Sex", options=["Female", "Male"])
        cp = st.selectbox(
            " Chest Pain Type (cp)", 
            options=[1, 2, 3, 4],
            help="1=Typical angina, 2=Atypical angina, 3=Non-anginal pain, 4=Asymptomatic"
        )
        trestbps = st.number_input(" Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120, step=1)
        chol = st.number_input(" Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)

    with col2:
        fbs = st.selectbox(" Fasting Blood Sugar > 120 mg/dl", options=[0, 1], help="1=True, 0=False")
        restecg = st.selectbox(" Resting ECG Results", options=[0, 1, 2])
        thalach = st.number_input(" Max Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
        exang = st.selectbox(" Exercise Induced Angina", options=[0, 1], help="1=Yes, 0=No")
        oldpeak = st.number_input(" ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    # Convert sex to numeric
    sex_value = 1 if sex == "Male" else 0

    if st.button(" Predict Diagnosis", type="primary", use_container_width=True):
        input_data = np.array([[
            age, sex_value, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak
        ]])
        
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        st.divider()
        
        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Diagnosis", "High Risk " if prediction == 1 else "Low Risk ")
        with col_res2:
            st.metric("Confidence", f"{probability:.2%}" if prediction == 1 else f"{(1-probability):.2%}")
        with col_res3:
            st.metric("Status", "Consult Doctor" if prediction == 1 else "Healthy")
        
        if prediction == 1:
            st.error(" **High Risk of Heart Disease detected.** Please consult a cardiologist.")
        else:
            st.success(" **Low Risk of Heart Disease.** Keep maintaining a healthy lifestyle!")

# -------------------------------------------------
# TAB 2: BATCH CSV UPLOAD
# -------------------------------------------------
with tab2:
    st.header("📁 Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV file containing patient data. The app will predict the risk for every row.")
    
    with st.expander(" View Required CSV Format"):
        st.markdown("Your CSV must contain these **exact 10 columns** (in any order):")
        cols_required = ", ".join(FEATURES)
        st.code(f"Required Columns: {cols_required}")
        
        sample_data = {
            "age": [50, 60, 45],
            "sex": [1, 0, 1],
            "cp": [2, 4, 1],
            "trestbps": [120, 140, 130],
            "chol": [200, 250, 180],
            "fbs": [0, 1, 0],
            "restecg": [0, 2, 1],
            "thalach": [150, 100, 170],
            "exang": [0, 1, 0],
            "oldpeak": [1.0, 2.5, 0.5]
        }
        df_sample = pd.DataFrame(sample_data)
        st.dataframe(df_sample, use_container_width=True)
        st.caption("Note: `sex`: 1=Male, 0=Female | `fbs`: 1=True, 0=False | `exang`: 1=Yes, 0=No")

    uploaded_file = st.file_uploader(
        " Drag and drop or browse for a CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="Upload a CSV with the required columns."
    )

    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            
            missing_cols = set(FEATURES) - set(df_input.columns)
            if missing_cols:
                st.error(f" Missing columns: {missing_cols}. Please check the CSV format.")
            else:
                with st.spinner(" Processing predictions..."):
                    time.sleep(1)
                    
                    X_input = df_input[FEATURES].copy()
                    X_input = X_input.fillna(0)
                    
                    X_scaled = scaler.transform(X_input)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]
                    
                    df_results = df_input.copy()
                    df_results['Risk_Score'] = probabilities
                    df_results['Prediction'] = predictions
                    df_results['Diagnosis'] = df_results['Prediction'].apply(
                        lambda x: "High Risk " if x == 1 else "Low Risk "
                    )
                    
                    st.divider()
                    st.subheader(" Prediction Summary")
                    
                    total_patients = len(df_results)
                    high_risk_count = int(sum(predictions == 1))
                    low_risk_count = total_patients - high_risk_count
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(" Total Patients", total_patients)
                    m2.metric(" High Risk", high_risk_count, delta=f"{high_risk_count/total_patients:.1%}" if total_patients > 0 else "0")
                    m3.metric(" Low Risk", low_risk_count, delta=f"{(low_risk_count/total_patients):.1%}" if total_patients > 0 else "0")
                    m4.metric(" Risk Percentage", f"{(high_risk_count/total_patients)*100:.1f}%")
                    
                    st.divider()
                    st.subheader(" Detailed Results")
                    
                    display_cols = FEATURES + ['Diagnosis', 'Risk_Score']
                    
                    # FIX: Remove the gradient to avoid matplotlib dependency
                    st.dataframe(
                        df_results[display_cols],
                        use_container_width=True,
                        height=400
                    )
                    
                    st.divider()
                    
                    csv_buffer = io.StringIO()
                    df_results.to_csv(csv_buffer, index=False)
                    csv_string = csv_buffer.getvalue()
                    
                    st.download_button(
                        label=" Download Results as CSV",
                        data=csv_string,
                        file_name="heart_disease_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )
                    
        except Exception as e:
            st.error(f" An error occurred: {e}")
            st.info("Please ensure your CSV is properly formatted.")

    else:
        st.info(" Upload a CSV file to begin batch predictions.")