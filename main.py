import streamlit as st
import xgboost as xgb
import pandas as pd
import pickle

# @st.cache_resource
def load_model():
    with open('xgb_model1.pkl', 'rb') as file:
        loaded_xgb_model = pickle.load(file)
    with open('randomforest_model.pkl', 'rb') as file:
        loaded_random_forest_model = pickle.load(file)
    with open('decision_tree_model.pkl', 'rb') as file:
        loaded_decision_tree_model = pickle.load(file)
    return loaded_xgb_model, loaded_decision_tree_model, \
        loaded_random_forest_model


st.set_page_config(layout="wide", page_title='Heart Disease Prediction')

st.sidebar.title("Heart Disease Prediction Application")
st.sidebar.write("")

xgb_model, decision_tree_model, random_forest_model = load_model()

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
        'Sex': ['M' if sex == 'Male' else 'F'],
        'ChestPainType': [chestPainType],
        'RestingBP': [int(restingBP)],
        'Cholesterol': [int(cholesterol)],
        'FastingBS': [0 if int(fastingBS) <= 120 else 1],
        'RestingECG': [restingECG],
        'MaxHR': [int(maxHR)],
        'ExerciseAngina': ['Y' if exerciseAngina == 'Yes' else 'N'],
        'Oldpeak': [float(oldpeak)],
        'ST_Slope': [st_slope]
    }

    df1 = pd.DataFrame(data)
    df = pd.concat([df, df1], axis=0)
    df = df.drop('HeartDisease', axis=1)
    ohe_variables = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina',
                     'ST_Slope']
    df = pd.get_dummies(data=df, prefix=ohe_variables, columns=ohe_variables)

    st.write(df.iloc[1:2])
    row_to_predict = df.iloc[1:2]

    data_to_predict = row_to_predict.values

    predictions = xgb_model.predict(xgb.DMatrix(data_to_predict))
    st.write(predictions)