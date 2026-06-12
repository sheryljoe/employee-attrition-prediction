# app.py  –  Employee Attrition Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
# ---- Page configuration ----
st.set_page_config(
page_title='Employee Attrition Predictor',
page_icon='\U0001f4ca',
layout='wide'
)
# ---- Load model and scaler ----
@st.cache_resource
def load_model():
    model  = joblib.load('random_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler
 
model, scaler = load_model()
# ---- App Title ----
st.title('\U0001f4ca Employee Attrition Prediction Dashboard')
st.write('Enter employee details below to predict attrition risk.')
st.divider()
# ---- Sidebar Input Form ----
st.sidebar.header('\U0001f464 Employee Details')
age           = st.sidebar.slider('Age', 18, 60, 35)
monthly_income = st.sidebar.number_input('Monthly Income (₹)', 1000, 20000, 5000, step=500)
overtime      = st.sidebar.selectbox('OverTime', ['No','Yes'])
job_sat       = st.sidebar.slider('Job Satisfaction (1=Low, 4=High)', 1, 4, 3)
env_sat       = st.sidebar.slider('Environment Satisfaction', 1, 4, 3)
work_life     = st.sidebar.slider('Work-Life Balance', 1, 4, 3)
years_company = st.sidebar.slider('Years at Company', 0, 40, 5)
distance      = st.sidebar.slider('Distance from Home (km)', 1, 30, 5)
job_level     = st.sidebar.selectbox('Job Level', [1, 2, 3, 4, 5])
num_companies = st.sidebar.slider('Number of Companies Worked', 0, 9, 2)
# Encode inputs to match training data
overtime_enc = 1 if overtime == 'Yes' else 0
# ---- Build input array (31 features – fill others with typical values) ----
input_data = pd.DataFrame([{
    'Age': age, 'BusinessTravel': 1, 'DailyRate': 800,
    'Department': 2, 'DistanceFromHome': distance,
    'Education': 3, 'EducationField': 3,
    'EmployeeCount': 1, 'EnvironmentSatisfaction': env_sat,
    'Gender': 1, 'HourlyRate': 65, 'JobInvolvement': 3,
    'JobLevel': job_level, 'JobRole': 5,
    'JobSatisfaction': job_sat, 'MaritalStatus': 1,
    'MonthlyIncome': monthly_income, 'MonthlyRate': 14000,
    'NumCompaniesWorked': num_companies, 'OverTime': overtime_enc,
    'PercentSalaryHike': 15, 'PerformanceRating': 3,
    'RelationshipSatisfaction': 3, 'StockOptionLevel': 1,
    'TotalWorkingYears': max(age-22, 1), 'TrainingTimesLastYear': 3,
    'WorkLifeBalance': work_life, 'YearsAtCompany': years_company,
    'YearsInCurrentRole': max(years_company-2, 0),
    'YearsSinceLastPromotion': 1, 'YearsWithCurrManager': 3,
}])
# ---- Predict ----
col1, col2 = st.columns(2)
with col1:
    st.subheader('\U0001f9e0 Attrition Prediction')
    if st.button('Predict Attrition Risk', type='primary'):
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1] * 100
        if prediction == 1:
            st.error(f'⚠️ HIGH RISK: This employee has a {probability:.1f}% probability of leaving.')
        else:
    st.success(f'✅ LOW RISK: This employee has only a {probability:.1f}% probability of leaving.')
with col2:
    st.subheader('\U0001f4cb Input Summary')
    summary = pd.DataFrame({'Attribute': ['Age','Monthly Income','OverTime','Job Satisfaction',
                                           'Work-Life Balance','Years at Company'],
                            'Value': [age, f'₹{monthly_income:,}', overtime,
                                      f'{job_sat}/4', f'{work_life}/4', years_company]})
    st.dataframe(summary, hide_index=True)
# BOTTOM of app.py
import matplotlib.pyplot as plt
import seaborn as sns
st.divider()
st.subheader('\U0001f4ca Attrition Analytics – Full Dataset Overview')
# Load original dataset for charts (include it in your GitHub repo)
try:
    raw_df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

    chart1, chart2 = st.columns(2)

    with chart1:
        fig, ax = plt.subplots(figsize=(6,4))
        dept_att = raw_df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()*100
        dept_att['Yes'].sort_values().plot(kind='barh', ax=ax, color='#C0392B')
        ax.set_title('Attrition Rate by Department (%)', fontweight='bold')
        ax.set_xlabel('Attrition Rate (%)')
        st.pyplot(fig)

    with chart2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.countplot(x='OverTime', hue='Attrition', data=raw_df,
                      palette=['#2E74B5','#C0392B'], ax=ax2)
        ax2.set_title('Overtime vs Attrition', fontweight='bold')
        st.pyplot(fig2)
except FileNotFoundError:
    st.info('Dataset file not found. Upload WA_Fn-UseC_-HR-Employee-Attrition.csv to your repo.')
