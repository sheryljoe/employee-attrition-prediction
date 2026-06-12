# app.py  –  Employee Attrition Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Page configuration ----
st.set_page_config(
    page_title='Employee Attrition Predictor',
    page_icon='📊',
    layout='wide'
)

# ---- Load model and scaler ----
@st.cache_resource
def load_model():
    model = joblib.load('random_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ---- Load dataset (used for employee lookup + analytics) ----
@st.cache_data
def load_data():
    try:
        return pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    except FileNotFoundError:
        return None

raw_df = load_data()

# ---- Encoding maps (MUST match what was used during model training) ----
department_map = {'Sales': 2, 'Research & Development': 1, 'Human Resources': 0}
job_role_map = {
    'Sales Executive': 7, 'Research Scientist': 6, 'Laboratory Technician': 2,
    'Manufacturing Director': 4, 'Healthcare Representative': 0, 'Manager': 3,
    'Sales Representative': 8, 'Research Director': 5, 'Human Resources': 1
}
business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 2, 'Travel_Frequently': 1}
gender_map = {'Female': 0, 'Male': 1}
marital_status_map = {'Divorced': 0, 'Married': 1, 'Single': 2}
education_field_map = {
    'Life Sciences': 1, 'Medical': 3, 'Marketing': 2,
    'Technical Degree': 4, 'Human Resources': 0, 'Other': 5
}

# ---- App Title ----
st.title('📊 Employee Attrition Prediction Dashboard')
st.write('Enter employee details below to predict attrition risk.')
st.divider()

# ============================================================
# SIDEBAR — Optional: pick an existing employee to pre-fill
# ============================================================
st.sidebar.header('👤 Employee Details')

selected_emp = None
if raw_df is not None and 'EmployeeNumber' in raw_df.columns:
    emp_options = ['-- Manual Entry --'] + raw_df['EmployeeNumber'].astype(str).tolist()
    emp_choice = st.sidebar.selectbox('Load Existing Employee (optional)', emp_options)
    if emp_choice != '-- Manual Entry --':
        selected_emp = raw_df[raw_df['EmployeeNumber'] == int(emp_choice)].iloc[0]

# Helper to get default value: from selected employee if available, else fallback
def default_val(col, fallback):
    if selected_emp is not None and col in selected_emp:
        return selected_emp[col]
    return fallback

# ---- Core inputs ----
age = st.sidebar.slider('Age', 18, 60, int(default_val('Age', 35)))
monthly_income = st.sidebar.number_input(
    'Monthly Income (₹)', 1000, 200000,
    int(default_val('MonthlyIncome', 5000)), step=500
)
overtime_default = default_val('OverTime', 'No')
overtime = st.sidebar.selectbox('OverTime', ['No', 'Yes'],
                                 index=['No', 'Yes'].index(overtime_default) if overtime_default in ['No','Yes'] else 0)
job_sat = st.sidebar.slider('Job Satisfaction (1=Low, 4=High)', 1, 4, int(default_val('JobSatisfaction', 3)))
env_sat = st.sidebar.slider('Environment Satisfaction', 1, 4, int(default_val('EnvironmentSatisfaction', 3)))
work_life = st.sidebar.slider('Work-Life Balance', 1, 4, int(default_val('WorkLifeBalance', 3)))
years_company = st.sidebar.slider('Years at Company', 0, 40, int(default_val('YearsAtCompany', 5)))
distance = st.sidebar.slider('Distance from Home (km)', 1, 30, int(default_val('DistanceFromHome', 5)))
job_level = st.sidebar.selectbox('Job Level', [1, 2, 3, 4, 5],
                                  index=int(default_val('JobLevel', 1)) - 1)
num_companies = st.sidebar.slider('Number of Companies Worked', 0, 9, int(default_val('NumCompaniesWorked', 2)))

# ---- Additional inputs requested: Department, Job Role, Years in Current Role ----
dept_default = default_val('Department', 'Sales')
department = st.sidebar.selectbox(
    'Department', list(department_map.keys()),
    index=list(department_map.keys()).index(dept_default) if dept_default in department_map else 0
)

role_default = default_val('JobRole', 'Sales Executive')
job_role = st.sidebar.selectbox(
    'Job Role', list(job_role_map.keys()),
    index=list(job_role_map.keys()).index(role_default) if role_default in job_role_map else 0
)

years_in_role = st.sidebar.slider('Years in Current Role', 0, 18, int(default_val('YearsInCurrentRole', 2)))

# ---- Encode categorical inputs ----
overtime_enc = 1 if overtime == 'Yes' else 0
department_enc = department_map[department]
job_role_enc = job_role_map[job_role]

# ---- Build input array (31 features) ----
input_data = pd.DataFrame([{
    'Age': age,
    'BusinessTravel': business_travel_map.get(default_val('BusinessTravel', 'Travel_Rarely'), 2),
    'DailyRate': int(default_val('DailyRate', 800)),
    'Department': department_enc,
    'DistanceFromHome': distance,
    'Education': int(default_val('Education', 3)),
    'EducationField': education_field_map.get(default_val('EducationField', 'Life Sciences'), 1),
    'EmployeeCount': 1,
    'EnvironmentSatisfaction': env_sat,
    'Gender': gender_map.get(default_val('Gender', 'Male'), 1),
    'HourlyRate': int(default_val('HourlyRate', 65)),
    'JobInvolvement': int(default_val('JobInvolvement', 3)),
    'JobLevel': job_level,
    'JobRole': job_role_enc,
    'JobSatisfaction': job_sat,
    'MaritalStatus': marital_status_map.get(default_val('MaritalStatus', 'Married'), 1),
    'MonthlyIncome': monthly_income,
    'MonthlyRate': int(default_val('MonthlyRate', 14000)),
    'NumCompaniesWorked': num_companies,
    'OverTime': overtime_enc,
    'PercentSalaryHike': int(default_val('PercentSalaryHike', 15)),
    'PerformanceRating': int(default_val('PerformanceRating', 3)),
    'RelationshipSatisfaction': int(default_val('RelationshipSatisfaction', 3)),
    'StockOptionLevel': int(default_val('StockOptionLevel', 1)),
    'TotalWorkingYears': int(default_val('TotalWorkingYears', max(age - 22, 1))),
    'TrainingTimesLastYear': int(default_val('TrainingTimesLastYear', 3)),
    'WorkLifeBalance': work_life,
    'YearsAtCompany': years_company,
    'YearsInCurrentRole': years_in_role,
    'YearsSinceLastPromotion': int(default_val('YearsSinceLastPromotion', 1)),
    'YearsWithCurrManager': int(default_val('YearsWithCurrManager', 3)),
}])

# ============================================================
# Prediction + Summary
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader('🧠 Attrition Prediction')
    if st.button('Predict Attrition Risk', type='primary'):
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100
        if prediction == 1:
            st.error(f'⚠️ HIGH RISK: This employee has a {probability:.1f}% probability of leaving.')
        else:
            st.success(f'✅ LOW RISK: This employee has only a {probability:.1f}% probability of leaving.')

with col2:
    st.subheader('📋 Input Summary')
    summary = pd.DataFrame({
        'Attribute': ['Age', 'Monthly Income', 'OverTime', 'Department', 'Job Role',
                       'Job Satisfaction', 'Environment Satisfaction', 'Work-Life Balance',
                       'Years at Company', 'Years in Current Role', 'Job Level',
                       'Number of Companies Worked', 'Distance from Home'],
        'Value': [age, f'₹{monthly_income:,}', overtime, department, job_role,
                  f'{job_sat}/4', f'{env_sat}/4', f'{work_life}/4',
                  years_company, years_in_role, job_level,
                  num_companies, f'{distance} km']
    })
    st.dataframe(summary, hide_index=True)

# ============================================================
# Analytics section
# ============================================================
st.divider()
st.subheader('📊 Attrition Analytics – Full Dataset Overview')

if raw_df is not None:
    chart1, chart2 = st.columns(2)

    with chart1:
        fig, ax = plt.subplots(figsize=(6, 4))
        dept_att = raw_df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack() * 100
        dept_att['Yes'].sort_values().plot(kind='barh', ax=ax, color='#C0392B')
        ax.set_title('Attrition Rate by Department (%)', fontweight='bold')
        ax.set_xlabel('Attrition Rate (%)')
        st.pyplot(fig)

    with chart2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.countplot(x='OverTime', hue='Attrition', data=raw_df,
                       palette=['#2E74B5', '#C0392B'], ax=ax2)
        ax2.set_title('Overtime vs Attrition', fontweight='bold')
        st.pyplot(fig2)
else:
    st.info('Dataset file not found. Upload WA_Fn-UseC_-HR-Employee-Attrition.csv to your repo.')
    # ---- Model Performance & Credits ----
st.divider()
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric('Model', 'Random Forest')
col_m2.metric('Accuracy', '~85%')
col_m3.metric('F1-Score', '~82%')
col_m4.metric('AUC-ROC', '~0.91')

st.caption(
    'Built by Sheryl Ann Joseph & Chaithrali Parabh | '
    'PGDM Business Analytics & MMS | '
    'DCyber TechLab Pvt Ltd | Aditya School of Business Management'
)
