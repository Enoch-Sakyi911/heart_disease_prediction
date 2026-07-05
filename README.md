# Heart Disease Prediction Dashboard

A Streamlit-based web application that predicts the likelihood of heart disease using a trained machine learning model. The app supports both single-patient predictions and batch predictions from CSV files.

## Overview

This project uses a logistic regression model trained on clinical health indicators to estimate heart disease risk. It provides a simple interface for entering patient data or uploading a CSV file with multiple records.

## Features

- Interactive single prediction form
- Batch CSV upload for multiple predictions
- Risk score and diagnosis output
- Downloadable prediction results as CSV
- Clean Streamlit dashboard with sidebar details

## Project Structure

```text
heart_disease_prediction/
├── app.py                  # Streamlit application
├── requirements.txt       # Python dependencies
├── test_data.csv          # Example test data
├── data/
│   └── heart-disease.names
├── models/
│   ├── heart_model.pkl    # Trained model
│   └── scaler.pkl        # Feature scaler
├── notebooks/
│   └── main_model.ipynb  # Model training notebook
└── README.md
```

## Required Model Inputs

The app expects the following 10 input features:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak

## Installation

1. Clone or open the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal in your browser.

## Usage

### Single Prediction

- Open the app in your browser.
- Fill in the patient details in the form.
- Click the prediction button to see the diagnosis and probability.

### Batch Prediction

- Go to the CSV upload tab.
- Upload a CSV file containing the required columns.
- The app will generate predictions for each row and allow you to download the results.

## Model Information

- Algorithm: Logistic Regression
- Reported Accuracy: 84.46%
- AUC Score: Approximately 0.90

## Notes

- The app depends on the trained model files in the models folder.
- If those files are missing, the app will not run correctly until the model is retrained or restored.

## License

This project is for educational and demonstration purposes.
