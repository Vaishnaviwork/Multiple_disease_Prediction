🩺 Multiple Disease Prediction System

Project Overview
This project is an AI-powered healthcare application that predicts the likelihood of Parkinson’s Disease, Kidney Disease, and Liver Disease based on patient inputs. It combines machine learning models with an interactive Streamlit interface to provide quick, accurate, and user-friendly predictions.

🚀 Features

Multi-Disease Prediction: Predicts Parkinson, Kidney, and Liver diseases in a single app.

Interactive Inputs: Sliders, dropdowns, and collapsible sections for a smooth user experience.

Probability Visualization: Shows risk level with dynamic progress bars and color-coded indicators.

User-Friendly Interface: Clean and professional Streamlit UI.

Portable and Scalable: Models saved with joblib for deployment anywhere.

🏗️ System Architecture

Frontend: Streamlit web interface for inputting patient data.
Backend: Python-based ML pipeline using scikit-learn.
Models Used: Logistic Regression, Random Forest, XGBoost (based on dataset).
Data Sources:

Parkinson’s Dataset

Kidney Disease Dataset

Indian Liver Patient Dataset

Workflow:

Input Data: User enters symptoms, lab results, or demographic info.

Data Preprocessing: Numeric scaling, categorical encoding, and missing value handling.

Model Prediction: ML models calculate probability of disease.

Output: Display predicted disease and risk level interactively.

📊 Datasets
Disease	Dataset	Features
Parkinson	parkinsons.csv	Voice measurements (Frequency, Jitter, Shimmer, etc.)
Kidney	kidney_disease.csv	Age, Blood Pressure, Albumin, Sugar, Blood test results
Liver	indian_liver_patient.csv	Age, Gender, Bilirubin levels, Liver enzymes, Proteins

Note: All datasets are preprocessed before model training. Missing values handled and categorical variables encoded.

🛠️ Tools and Technologies

Programming Language: Python

Libraries: pandas, numpy, scikit-learn, joblib, streamlit

IDE: Visual Studio Code / Jupyter Notebook

Deployment: Streamlit local or cloud deployment

💻 Installation

Clone the repository

git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>


Create a virtual environment

python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py

📈 Usage

Select a disease from the sidebar.

Fill in numeric and categorical values interactively.

Click Predict to get the disease prediction and probability.

Visual risk indicator shows Low, Medium, or High risk.

📂 Project Structure
multiple_disease_prediction/
│
├─ src/
│  ├─ train_parkinsons.py
│  ├─ train_kidney.py
│  ├─ train_liver.py
│  ├─ data_preprocessing.py
│  └─ model_registry/
│       ├─ parkinsons_status.joblib
│       ├─ kidney_classification.joblib
│       └─ liver_disease_classifier.joblib
│
├─ datasets/
│  ├─ parkinsons.csv
│  ├─ kidney_disease.csv
│  └─ indian_liver_patient.csv
│
├─ app.py
├─ requirements.txt
└─ README.md

📌 Key Highlights

Fully interactive and polished UI.

Slider & dropdowns ensure realistic input ranges.

Color-coded risk visualization improves interpretability.

Easy to extend for more diseases.
