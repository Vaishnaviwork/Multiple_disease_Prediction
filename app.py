import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Load trained models
# -----------------------------
parkinsons_model = joblib.load("src/model_registry/parkinsons_status.joblib")
kidney_model = joblib.load("src/model_registry/kidney_classification.joblib")
liver_model = joblib.load("src/model_registry/liver_disease_classifier.joblib")

# -----------------------------
# App title
# -----------------------------
st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")
st.title("🩺 Multiple Disease Prediction System")
st.markdown("Predict Parkinson, Kidney, and Liver diseases using interactive inputs and trained ML models.")

# -----------------------------
# Sidebar for disease selection
# -----------------------------
disease_choice = st.sidebar.selectbox(
    "Select Disease",
    ["Parkinson", "Kidney", "Liver"]
)

# -----------------------------
# Helper function to show probability bar
# -----------------------------
def show_probability(proba):
    st.progress(proba)
    if proba >= 0.75:
        st.success(f"High Risk ({proba*100:.2f}%)")
    elif proba >= 0.5:
        st.warning(f"Medium Risk ({proba*100:.2f}%)")
    else:
        st.info(f"Low Risk ({proba*100:.2f}%)")

# -----------------------------
# Parkinson Prediction
# -----------------------------
if disease_choice == "Parkinson":
    st.header("Parkinson Disease Prediction")
    st.markdown("Enter voice measurement values:")

    # Feature groups
    frequency_features = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)']
    jitter_features = ['MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP']
    shimmer_features = ['MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA']
    other_features = ['NHR','HNR','RPDE','DFA','spread1','spread2','D2','PPE']

    input_values = []

    with st.expander("Frequency Features"):
        for feature in frequency_features:
            val = st.number_input(feature, value=0.0, format="%.5f")
            input_values.append(val)

    with st.expander("Jitter Features"):
        for feature in jitter_features:
            val = st.number_input(feature, value=0.0, format="%.5f")
            input_values.append(val)

    with st.expander("Shimmer Features"):
        for feature in shimmer_features:
            val = st.number_input(feature, value=0.0, format="%.5f")
            input_values.append(val)

    with st.expander("Other Features"):
        for feature in other_features:
            val = st.number_input(feature, value=0.0, format="%.5f")
            input_values.append(val)

    numeric_features = frequency_features + jitter_features + shimmer_features + other_features
    input_df = pd.DataFrame([input_values], columns=numeric_features)

    if st.button("Predict Parkinson"):
        prediction = parkinsons_model.predict(input_df)[0]
        proba = parkinsons_model.predict_proba(input_df)[0][1] if hasattr(parkinsons_model, "predict_proba") else None
        st.success(f"Prediction: {'Parkinson Disease' if prediction==1 else 'Healthy'}")
        if proba is not None:
            show_probability(proba)

# -----------------------------
# Kidney Prediction
# -----------------------------
elif disease_choice == "Kidney":
    st.header("Kidney Disease Prediction")
    st.markdown("Enter patient details:")

    # Numeric features with sliders
    st.subheader("Numeric Features")
    age = st.slider("Age", 1, 120, 30)
    bp = st.slider("Blood Pressure (bp)", 60, 200, 80)
    sg = st.selectbox("Specific Gravity (sg)", ["1.005","1.010","1.015","1.020","1.025"])
    al = st.selectbox("Albumin (al)", [0,1,2,3,4,5])
    su = st.selectbox("Sugar (su)", [0,1,2,3,4,5])
    bgr = st.slider("Blood Glucose Random (bgr)", 50, 500, 80)
    bu = st.slider("Blood Urea (bu)", 5, 200, 20)
    sc = st.slider("Serum Creatinine (sc)", 0.1, 20.0, 1.0)
    sod = st.slider("Sodium (sod)", 100, 180, 135)
    pot = st.slider("Potassium (pot)", 2, 10, 4)
    hemo = st.slider("Hemoglobin (hemo)", 5, 25, 15)
    pcv = st.slider("PCV", 10, 60, 40)
    wc = st.slider("White Blood Cells (wc)", 2000, 20000, 8000)
    rc = st.slider("Red Blood Cells (rc)", 2, 7, 4)

    numeric_values = [age,bp,sg,al,su,bgr,bu,sc,sod,pot,hemo,pcv,wc,rc]

    # Categorical features
    st.subheader("Categorical Features")
    rbc = st.selectbox("Red Blood Cells (rbc)", ["yes","no"])
    pc = st.selectbox("Pus Cell (pc)", ["yes","no"])
    pcc = st.selectbox("Pus Cell Clumps (pcc)", ["yes","no"])
    ba = st.selectbox("Bacteria (ba)", ["yes","no"])
    htn = st.selectbox("Hypertension (htn)", ["yes","no"])
    dm = st.selectbox("Diabetes Mellitus (dm)", ["yes","no"])
    cad = st.selectbox("Coronary Artery Disease (cad)", ["yes","no"])
    appet = st.selectbox("Appetite (appet)", ["good","poor"])
    pe = st.selectbox("Pedal Edema (pe)", ["yes","no"])
    ane = st.selectbox("Anemia (ane)", ["yes","no"])

    cat_values = [rbc, pc, pcc, ba, htn, dm, cad, appet, pe, ane]

    input_df = pd.DataFrame([numeric_values + cat_values], 
                            columns=['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wc','rc'] + 
                                    ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'])

    if st.button("Predict Kidney"):
        prediction = kidney_model.predict(input_df)[0]
        proba = kidney_model.predict_proba(input_df)[0][1] if hasattr(kidney_model, "predict_proba") else None
        st.success(f"Prediction: {'Chronic Kidney Disease' if prediction=='ckd' else 'Healthy'}")
        if proba is not None:
            show_probability(proba)

# -----------------------------
# Liver Prediction
# -----------------------------
elif disease_choice == "Liver":
    st.header("Liver Disease Prediction")
    st.markdown("Enter patient details:")

    st.subheader("Numeric Features")
    Age = st.slider("Age", 1, 100, 30)
    Total_Bilirubin = st.slider("Total Bilirubin", 0.1, 50.0, 1.0)
    Direct_Bilirubin = st.slider("Direct Bilirubin", 0.0, 30.0, 0.1)
    Alkaline_Phosphotase = st.slider("Alkaline Phosphotase", 0, 500, 100)
    Alamine_Aminotransferase = st.slider("Alamine Aminotransferase", 0, 300, 20)
    Aspartate_Aminotransferase = st.slider("Aspartate Aminotransferase", 0, 300, 20)
    Total_Protiens = st.slider("Total Proteins", 0.0, 10.0, 6.0)
    Albumin = st.slider("Albumin", 0.0, 5.0, 3.5)
    Albumin_and_Globulin_Ratio = st.slider("Albumin and Globulin Ratio", 0.0, 2.5, 1.0)

    numeric_values = [Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,
                      Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,
                      Albumin,Albumin_and_Globulin_Ratio]

    st.subheader("Categorical Features")
    Gender = st.selectbox("Gender", ["Male","Female"])

    input_df = pd.DataFrame([numeric_values + [Gender]],
                            columns=['Age','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase',
                                     'Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens',
                                     'Albumin','Albumin_and_Globulin_Ratio','Gender'])

    if st.button("Predict Liver"):
        prediction = liver_model.predict(input_df)[0]
        proba = liver_model.predict_proba(input_df)[0][1] if hasattr(liver_model, "predict_proba") else None
        st.success(f"Prediction: {'Liver Disease' if prediction==1 else 'Healthy'}")
        if proba is not None:
            show_probability(proba)
