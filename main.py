import streamlit as st
import pandas as pd
import pickle


@st.cache_resource
def load_model():
    with open('xgb_model1.pkl', 'rb') as file:
        loaded_xgb_model = pickle.load(file)

    return loaded_xgb_model


st.set_page_config(layout="wide", page_title='Heart Disease Prediction')


st.sidebar.title("Heart Disease Prediction Application")
with st.sidebar.expander("About"):
    st.write(f"The Movie Recommender uses cosine similarity to suggest "
             f"movies based on user input. The system "
             f"is built using TMDB's 5000 movie dataset. Additional "
             f"information is "
             f"retrieved from TMDB's API."
             f" This project was initiated for a course at my university"
             f" and is still a work in progress. If you would like to give"
             f" feedback or contribute, the source code and documentation "
             f"for the project can be found "
             f"[here](https://github.com/YohanV1/TheMovieRecommender)."
             f" If you have any suggestions or questions, "
             f"please don't hesitate to reach out.")


xgb_model = load_model()

st.header('Heart Disease Prediction - Decision Trees, '
          'Random Forest, and XGBoost')

col1, ecol, col2 = st.columns([1.5, 0.2, 1.5])

with col1:
    age = st.text_input('Age (in years):')
    sex = st.selectbox('Sex:', ('', 'Male', 'Female'))
    chestPainType = st.selectbox('Chest Pain Type:', ('', 'TA', 'ATA', 'NAP',
                                                      'ASY'),
                                 help='TA: Typical Angina, '
                                      'ATA: Atypical Angina, '
                                      'NAP: Non-Anginal Pain, '
                                      'ASY: Asymptomatic')
    restingBP = st.text_input('Resting Blood Pressure [mm Hg]: ')
    cholesterol = st.text_input('Serum Cholesterol [mg/dL]: ')
    fastingBS = st.text_input('Fasting Blood Sugar [mg/dL]: ')

with col2:
    restingECG = st.selectbox('Resting Electrocardiogram Results: ',
                              ('', 'Normal', 'ST', 'LVH'),
                              help = "ST: having ST-T wave abnormality "
                                     "(T wave inversions "
                                     "and/or ST elevation or "
                                     "depression of > 0.05 mV), LVH: showing "
                                     "probable or definite left ventricular "
                                     "hypertrophy by Estes' criteria")
    maxHR = st.text_input('Maximum Heart Rate Achieved: ')
    exerciseAngina = st.selectbox('Exercise-induced Angina: ',
                                  ('', 'Yes', 'No'))
    oldpeak = st.text_input('Oldpeak: oldpeak = ST '
                            '[Numeric value measured in depression]')
    st_slope = st.selectbox('ST Slope (the slope of the peak exercise ST '
                            'segment): ', ('', 'Up',
                                           'Flat', 'Down'))

b = st.button("Run Model.")

if b:
    if age == '' or restingBP == '' or cholesterol == '' or fastingBS \
            == '' or maxHR == '' or oldpeak == '':
        st.error('Please specify all the details.')
        print(age, restingBP, cholesterol, fastingBS, maxHR, oldpeak)
    elif sex == '':
        st.error('Please specify sex.')
    elif chestPainType == '':
        st.error('Please specify chest pain type.')
    elif restingECG == '':
        st.error('Please specify ECG results.')
    elif exerciseAngina == '':
        st.error('Please specify if you have exercise-induced angina.')
    elif st_slope == '':
        st.error('Please specify ST slope.')
    data = {
        'Age': [int(age)],
        'RestingBP': [int(restingBP)],
        'Cholesterol': [int(cholesterol)],
        'FastingBS': [0 if int(fastingBS) <= 120 else 1],
        'MaxHR': [int(maxHR)],
        'Oldpeak': [float(oldpeak)],
        'Sex_F': [1 if sex == 'Female' else 0],
        'Sex_M': [1 if sex == 'Male' else 0],
        'ChestPainType_ASY': [1 if chestPainType == "ASY" else 0],
        'ChestPainType_ATA': [1 if chestPainType == "ATA" else 0],
        'ChestPainType_NAP': [1 if chestPainType == "NAP" else 0],
        'ChestPainType_TA': [1 if chestPainType == "TA" else 0],
        'RestingECG_LVH': [1 if restingECG == "LVH" else 0],
        'RestingECG_Normal': [1 if restingECG == "Normal" else 0],
        'RestingECG_ST': [1 if restingECG == "ST" else 0],
        'ExerciseAngina_N': [0 if exerciseAngina == 'Yes' else 1],
        'ExerciseAngina_Y': [0 if exerciseAngina == 'No' else 1],
        'ST_Slope_Down': [1 if st_slope == "Down" else 0],
        'ST_Slope_Flat': [1 if st_slope == "Flat" else 0],
        'ST_Slope_Up': [1 if st_slope == "Up" else 0]
    }

    df = pd.DataFrame(data)

    prediction = xgb_model.predict(df.iloc[0:1])

    if prediction[0] == 0:
        st.success('Patient does not have heart disease.'
                   '\n XGBoost Model Metrics:'
                   '\n Train Accuracy = 95.78%'
                   '\n Test Accuracy = 90.76%')
    if prediction[0] == 1:
        st.error('Patient has heart disease.'
                   '\n XGBoost Model Metrics:'
                   '\n Train Accuracy = 95.78%'
                   '\n Test Accuracy = 90.76%')