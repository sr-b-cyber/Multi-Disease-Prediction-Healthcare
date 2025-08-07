import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# Set page config
st.set_page_config(page_title="Multi-Disease Prediction App", layout="wide", page_icon="🧬")

# Utility: Load models safely using joblib
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {str(e)}")
        return None

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load models & preprocessors with error handling
Diabetes_model = load_model(f'{working_dir}/models/Diabetes_model.pkl')
Asthma_model = load_model(f'{working_dir}/models/Asthma_model.pkl')
BP_model = load_model(f'{working_dir}/models/BP_model.pkl')
Typhoid_model = load_model(f'{working_dir}/models/Typhoid_model.pkl')

Diabetes_preprocessor = load_model(f'{working_dir}/models/Diabetes_preprocessor.pkl')
Asthma_preprocessor = load_model(f'{working_dir}/models/Asthma_preprocessor.pkl')
BP_preprocessor = load_model(f'{working_dir}/models/BP_preprocessor.pkl')

# Load Typhoid preprocessors separately
Typhoid_scaler = load_model(f'{working_dir}/models/Typhoid_scaler.pkl')
Typhoid_encoder = load_model(f'{working_dir}/models/Typhoid_encoder.pkl')
Typhoid_num_imputer = load_model(f'{working_dir}/models/Typhoid_num_imputer.pkl')
Typhoid_cat_imputer = load_model(f'{working_dir}/models/Typhoid_cat_imputer.pkl')

# Sidebar Navigation
with st.sidebar:
    selected = option_menu("Disease Prediction System",
        ["Diabetes Prediction", "Asthma Prediction", "Blood Pressure Prediction", "Typhoid Prediction"],
        icons=['activity', 'wind', 'heart-pulse', 'thermometer'],
        default_index=0)

# ===================== Diabetes Page =====================
if selected == "Diabetes Prediction":
    st.title("🔍 Diabetes Prediction")

    col1, col2 = st.columns(2)
    with col1:
        HighBP = st.radio("High Blood Pressure", [0, 1])
        HighChol = st.radio("High Cholesterol", [0, 1])
        CholCheck = st.radio("Cholesterol Check in 5 Years", [0, 1])
        BMI = st.number_input("BMI", 0.0, 70.0)
        Smoker = st.radio("Smoker", [0, 1])
        Stroke = st.radio("Ever had Stroke", [0, 1])
        HeartDiseaseorAttack = st.radio("Heart Disease or Attack", [0, 1])
        PhysActivity = st.radio("Physical Activity", [0, 1])
        Fruits = st.radio("Consume Fruits", [0, 1])
        Veggies = st.radio("Consume Vegetables", [0, 1])
        HvyAlcoholConsump = st.radio("Heavy Alcohol Consumption", [0, 1])
        AnyHealthcare = st.radio("Any Healthcare", [0, 1])
        NoDocbcCost = st.radio("No Doctor due to Cost", [0, 1])
        GenHlth = st.radio("General Health (1=Excellent to 5=Poor)", [1, 2, 3, 4, 5])
        MentHlth = st.number_input("Mental Health (days)", 0, 30)
        PhysHlth = st.number_input("Physical Health (days)", 0, 30)
        DiffWalk = st.radio("Difficulty Walking", [0, 1])
        Sex = st.radio("Sex (0=Female, 1=Male)", [0, 1])
        Age = st.number_input("Age", 1, 120)

    if st.button("Predict Diabetes"):
        try:
            # Raw input as dict
            input_dict = {
                'HighBP': HighBP, 'HighChol': HighChol, 'CholCheck': CholCheck, 'BMI': BMI,
                'Smoker': Smoker, 'Stroke': Stroke, 'HeartDiseaseorAttack': HeartDiseaseorAttack,
                'PhysActivity': PhysActivity, 'Fruits': Fruits, 'Veggies': Veggies,
                'HvyAlcoholConsump': HvyAlcoholConsump, 'AnyHealthcare': AnyHealthcare,
                'NoDocbcCost': NoDocbcCost, 'GenHlth': GenHlth, 'MentHlth': MentHlth,
                'PhysHlth': PhysHlth, 'DiffWalk': DiffWalk, 'Sex': Sex, 'Age': Age
            }

            # Define numeric columns used in StandardScaler
            num_cols = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age']
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Split and transform
            X_scaled = Diabetes_preprocessor.transform(input_df[num_cols])
            X_other = input_df.drop(columns=num_cols).values
            
            # Concatenate final input
            X_final = np.concatenate([X_other, X_scaled], axis=1)
            
            # Predict
            result = Diabetes_model.predict(X_final)[0]
            st.success("✅ Positive for Diabetes" if result else "❌ Negative for Diabetes")
        except Exception as e:
            st.error(f"Prediction error: {e}")




elif selected == "Asthma Prediction":
    st.title("🌬 Asthma Prediction")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 1, 120)
        BMI = st.number_input("BMI", 0.0, 70.0)
        Smoking = st.radio("Smoking (0=No, 1=Yes)", [0, 1])
        PhysicalActivity = st.slider("Physical Activity", 0, 10)
        DietQuality = st.slider("Diet Quality", 0, 10)
        SleepQuality = st.slider("Sleep Quality", 0, 10)
        PollutionExposure = st.slider("Pollution Exposure", 0, 10)
        PollenExposure = st.slider("Pollen Exposure", 0, 10)
        DustExposure = st.slider("Dust Exposure", 0, 10)
        PetAllergy = st.radio("Pet Allergy (0=No, 1=Yes)", [0, 1])
        FamilyHistoryAsthma = st.radio("Family History of Asthma (0=No, 1=Yes)", [0, 1])
        HistoryOfAllergies = st.radio("History of Allergies (0=No, 1=Yes)", [0, 1])

    with col2:
        Eczema = st.radio("Eczema (0=No, 1=Yes)", [0, 1])
        HayFever = st.radio("Hay Fever (0=No, 1=Yes)", [0, 1])
        GastroesophagealReflux = st.radio("Gastroesophageal Reflux (0=No, 1=Yes)", [0, 1])
        LungFunctionFEV1 = st.number_input("Lung Function FEV1", 0.0, 10.0)
        LungFunctionFVC = st.number_input("Lung Function FVC", 0.0, 10.0)
        Wheezing = st.radio("Wheezing (0=No, 1=Yes)", [0, 1])
        ShortnessOfBreath = st.radio("Shortness of Breath (0=No, 1=Yes)", [0, 1])
        ChestTightness = st.radio("Chest Tightness (0=No, 1=Yes)", [0, 1])
        Coughing = st.radio("Coughing (0=No, 1=Yes)", [0, 1])
        NighttimeSymptoms = st.radio("Nighttime Symptoms (0=No, 1=Yes)", [0, 1])
        ExerciseInduced = st.radio("Exercise Induced Symptoms (0=No, 1=Yes)", [0, 1])

    if st.button("Predict Asthma"):
        try:
            # Define feature order exactly as in training data
            feature_order = [
                'Age', 'BMI', 'Smoking', 'PhysicalActivity', 'DietQuality',
                'SleepQuality', 'PollutionExposure', 'PollenExposure', 'DustExposure',
                'PetAllergy', 'FamilyHistoryAsthma', 'HistoryOfAllergies',
                'Eczema', 'HayFever', 'GastroesophagealReflux',
                'LungFunctionFEV1', 'LungFunctionFVC',
                'Wheezing', 'ShortnessOfBreath', 'ChestTightness',
                'Coughing', 'NighttimeSymptoms', 'ExerciseInduced'
            ]

            input_data = {
                'Age': Age,
                'BMI': BMI,
                'Smoking': Smoking,
                'PhysicalActivity': PhysicalActivity,
                'DietQuality': DietQuality,
                'SleepQuality': SleepQuality,
                'PollutionExposure': PollutionExposure,
                'PollenExposure': PollenExposure,
                'DustExposure': DustExposure,
                'PetAllergy': PetAllergy,
                'FamilyHistoryAsthma': FamilyHistoryAsthma,
                'HistoryOfAllergies': HistoryOfAllergies,
                'Eczema': Eczema,
                'HayFever': HayFever,
                'GastroesophagealReflux': GastroesophagealReflux,
                'LungFunctionFEV1': LungFunctionFEV1,
                'LungFunctionFVC': LungFunctionFVC,
                'Wheezing': Wheezing,
                'ShortnessOfBreath': ShortnessOfBreath,
                'ChestTightness': ChestTightness,
                'Coughing': Coughing,
                'NighttimeSymptoms': NighttimeSymptoms,
                'ExerciseInduced': ExerciseInduced
            }

            input_df = pd.DataFrame([input_data])[feature_order]

            # Define numeric features in exact order used during training
            numeric_features = [
                'BMI', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'Age',
                'PollutionExposure', 'PollenExposure', 'DustExposure',
                'LungFunctionFEV1', 'LungFunctionFVC'
            ]
            categorical_features = [col for col in feature_order if col not in numeric_features]

            # Correct transformation with column names
            scaled_df = pd.DataFrame(
                Asthma_preprocessor.transform(input_df[numeric_features]),
                columns=numeric_features
            )

            categorical_df = input_df[categorical_features].reset_index(drop=True)

            # Reorder columns to match training data order
            final_columns = feature_order
            final_input_df = pd.concat([scaled_df, categorical_df], axis=1)[final_columns]

            prediction = Asthma_model.predict(final_input_df)[0]

            st.success("✅ Asthma Detected" if prediction else "❌ No Asthma Detected")

        except Exception as e:
            st.error(f"Prediction error: {e}")



# ===================== Blood Pressure Page =====================
elif selected == "Blood Pressure Prediction":
    st.title("🩸 Blood Pressure Prediction")

    col1, col2 = st.columns(2)
    with col1:
        Age = st.number_input("Age", 1, 120)
        BMI = st.number_input("BMI", 0.0, 70.0)
        Sex = st.radio("Sex (0=Female, 1=Male)", [0, 1])
        Pregnancy = st.number_input("Pregnancy Status (0=No, 1=Yes)", 0, 1)
        Smoking = st.radio("Smoking Status", [0, 1])
        PhysicalActivity = st.number_input("Physical Activity (steps/day)", 0, 50000)
        SaltIntake = st.number_input("Salt Intake (mg)", 0, 50000)
        Alcohol = st.number_input("Alcohol Consumption (ml/day)", 0, 1000)
        Stress = st.selectbox("Stress Level", [1, 2, 3])
        Level_of_Hemoglobin = st.number_input("Level of Hemoglobin (g/dL)", 0.0, 25.0)
        Genetic_Pedigree_Coefficient = st.number_input("Genetic Pedigree Coefficient", 0.0, 2.5)
        Chronic_kidney_disease = st.radio("Chronic Kidney Disease", [0, 1])
        AdrenalThyroid = st.radio("Adrenal and Thyroid Disorders", [0, 1])

    if st.button("Predict Blood Pressure"):
        try:
            input_data = {
                'Level_of_Hemoglobin': Level_of_Hemoglobin,
                'Genetic_Pedigree_Coefficient': Genetic_Pedigree_Coefficient,
                'Age': Age,
                'BMI': BMI,
                'Sex': Sex,
                'Pregnancy': Pregnancy,
                'Smoking': Smoking,
                'Physical_activity': PhysicalActivity,
                'salt_content_in_the_diet': SaltIntake,
                'alcohol_consumption_per_day': Alcohol,
                'Level_of_Stress': Stress,
                'Chronic_kidney_disease': Chronic_kidney_disease,
                'Adrenal_and_thyroid_disorders': AdrenalThyroid
            }

            # Create DataFrame from user input
            df_input = pd.DataFrame([input_data])

            # Define columns to scale
            cols_to_scale = [
                'Level_of_Hemoglobin', 'Genetic_Pedigree_Coefficient',
                'Age', 'BMI', 'Physical_activity',
                'salt_content_in_the_diet', 'alcohol_consumption_per_day',
                'Level_of_Stress'
            ]

            # Scale the numeric columns
            X_scaled = BP_preprocessor.transform(df_input[cols_to_scale])
            
            # Get the unscaled columns
            X_unscaled = df_input.drop(columns=cols_to_scale).values

            # Concatenate scaled and unscaled features
            X_final = np.concatenate([X_scaled, X_unscaled], axis=1)

            # Predict using all 13 features
            result = BP_model.predict(X_final)[0]

            st.success("✅ High Blood Pressure Detected" if result else "❌ Normal Blood Pressure")

        except Exception as e:
            st.error(f"Prediction error: {e}")

# ===================== Typhoid Page =====================
elif selected == "Typhoid Prediction":
    st.title("🦠 Typhoid Disease Prediction")

    try:
        Location = st.selectbox("Location", ["Urban", "Rural", "Endemic"])
        SES = st.selectbox("Socioeconomic Status", ["Low", "Middle", "High"])
        WaterSource = st.selectbox("Water Source", ["Tap", "Untreated Supply", "River", "Well"])
        Sanitation = st.selectbox("Sanitation Facilities", ["Proper", "Open Defecation"])
        HandHygiene = st.radio("Hand Hygiene", [0, 1])
        StreetFood = st.radio("Consumption of Street Food", [0, 1])
        FeverDays = st.number_input("Fever Duration (days)", 0, 30)
        SkinIssues = st.radio("Skin Manifestations", [0, 1])
        WBC = st.number_input("White Blood Cell Count", 0, 20000)
        Platelets = st.number_input("Platelet Count", 0, 1000000)
        CultureResult = st.radio("Blood Culture Result", [0, 1])
        TyphoidTest = st.selectbox("Typhoid Test", ["IgG Positive", "Negative"])
        Vaccination = st.selectbox("Vaccination Status", ["Received", "Not Received"])
        History = st.selectbox("Previous Typhoid History", ["Yes", "No"])
        Weather = st.selectbox("Weather Conditions", ["Hot & Dry", "Rainy & Wet", "Moderate", "Cold & Humid"])

        if st.button("Predict Typhoid"):
            # Check if models are available
            if Typhoid_model is None or Typhoid_scaler is None or Typhoid_encoder is None:
                st.warning("⚠️ Typhoid prediction model is currently unavailable. Please try again later.")
            else:
                try:
                    # Define categorical and numerical columns
                    categorical_cols = ['Previous History of Typhoid','Typhoid Vaccination Status','Blood Culture Result',
                                       'Skin Manifestations','Sanitation Facilities','Hand Hygiene','Consumption of Street Food',
                                       'Location', 'Socioeconomic Status', 'Water Source Type', 'Typhidot Test','Weather Condition']
                    numerical_cols = ['Fever Duration (Days)', 'White Blood Cell Count', 'Platelet Count']
                    
                    # Create input data
                    input_data = pd.DataFrame([{
                        'Location': Location,
                        'Socioeconomic Status': SES,
                        'Water Source Type': WaterSource,
                        'Sanitation Facilities': Sanitation,
                        'Hand Hygiene': HandHygiene,
                        'Consumption of Street Food': StreetFood,
                        'Fever Duration (Days)': FeverDays,
                        'Skin Manifestations': SkinIssues,
                        'White Blood Cell Count': WBC,
                        'Platelet Count': Platelets,
                        'Blood Culture Result': CultureResult,
                        'Typhidot Test': TyphoidTest,
                        'Typhoid Vaccination Status': Vaccination,
                        'Previous History of Typhoid': History,
                        'Weather Condition': Weather
                    }])
                    
                    # Apply manual preprocessing (same as in notebook)
                    # Process numerical columns
                    input_num = Typhoid_num_imputer.transform(input_data[numerical_cols])
                    input_num_scaled = Typhoid_scaler.transform(input_num)
                    
                    # Process categorical columns
                    input_cat = Typhoid_cat_imputer.transform(input_data[categorical_cols])
                    input_cat_encoded = Typhoid_encoder.transform(input_cat)
                    
                    # Combine numerical and categorical features
                    input_processed = np.hstack([input_num_scaled, input_cat_encoded])
                    
                    # Make prediction
                    result = Typhoid_model.predict(input_processed)[0]
                    typhoid_classes = {
                        0: "No Typhoid",
                        1: "Relapsing Typhoid",
                        2: "Complicated Typhoid",
                        3: "Acute Typhoid Fever"
                    }
                    st.success(f"🦠 Prediction: {typhoid_classes.get(result, 'Unknown')}")
                except Exception as e:
                    st.error(f"Typhoid Prediction error: {e}")
    except Exception as e:
        st.error(f"Typhoid Prediction error: {e}")