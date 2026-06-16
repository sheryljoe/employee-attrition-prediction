# app.py  –  Employee Attrition Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title='Employee Attrition Predictor',
    page_icon='🏢',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

/* ── Hide default Streamlit chrome ── */
#MainMenu        {visibility: hidden;}
footer           {visibility: hidden;}
header           {visibility: hidden;}

/* ── Global font & background ── */
html, body, [class*="css"] {
    font-family: 'Segoe UI', Arial, sans-serif;
}
.main {background-color: #F4F6FA;}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1F3864 0%, #2E74B5 100%);
    padding-top: 10px;
}
[data-testid="stSidebar"] * {color: #E8F0FE !important;}
[data-testid="stSidebar"] .stSlider > label,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stNumberInput > label {
    color: #BDD7EE !important;
    font-size: 13px;
    font-weight: 500;
}
[data-testid="stSidebar"] hr {border-color: rgba(255,255,255,0.2);}

/* ── Name Logo ── */
.name-logo {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 0 14px 0;
}
.logo-circle {
    width: 56px; height: 56px;
    border-radius: 50%;
    background: linear-gradient(135deg, #1F3864, #2E74B5);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 800;
    color: white; letter-spacing: 1px;
    box-shadow: 0 2px 8px rgba(31,56,100,0.25);
    flex-shrink: 0;
}
.logo-text-block {line-height: 1.3;}
.logo-title {
    font-size: 22px; font-weight: 700;
    color: #1F3864; letter-spacing: -0.3px;
}
.logo-sub {font-size: 12.5px; color: #555; margin-top: 1px;}

/* ── Divider header ── */
.section-header {
    font-size: 15px; font-weight: 600;
    color: #1F3864; margin: 6px 0 10px 0;
    border-left: 4px solid #2E74B5;
    padding-left: 10px;
}

/* ── Prediction result cards ── */
.card-high {
    background: #FAECE7;
    border-left: 6px solid #C0392B;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0;
}
.card-high h3 {color: #C0392B; margin: 0 0 4px 0; font-size: 17px;}
.card-high h2 {color: #922B21; margin: 0 0 8px 0; font-size: 26px;}
.card-high p  {color: #641E16; font-size: 13px; margin: 0;}

.card-low {
    background: #EAF4E8;
    border-left: 6px solid #27AE60;
    border-radius: 10px;
    padding: 18px 22px;
    margin: 8px 0;
}
.card-low h3 {color: #1E8449; margin: 0 0 4px 0; font-size: 17px;}
.card-low h2 {color: #145A32; margin: 0 0 8px 0; font-size: 26px;}
.card-low p  {color: #0B5345; font-size: 13px; margin: 0;}

/* ── Metric footer cards ── */
.metric-row {
    display: flex; gap: 14px; margin-top: 6px;
}
.metric-box {
    flex: 1;
    background: white;
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.08);
    text-align: center;
    border-top: 3px solid #2E74B5;
}
.metric-box .m-label {font-size: 11px; color: #888; font-weight: 600;
                       text-transform: uppercase; letter-spacing: .05em;}
.metric-box .m-value {font-size: 22px; font-weight: 700; color: #1F3864; margin-top: 4px;}
.metric-box .m-sub   {font-size: 11px; color: #aaa; margin-top: 2px;}

/* ── Summary table ── */
.summary-table {width:100%; border-collapse:collapse; font-size:13px;}
.summary-table th {
    background:#1F3864; color:white;
    padding:8px 12px; text-align:left; font-weight:600;
}
.summary-table td {
    padding:7px 12px;
    border-bottom:1px solid #E8ECF2;
    color:#333;
}
.summary-table tr:nth-child(even) td {background:#F4F6FA;}

/* ── Sidebar predict button ── */
[data-testid="stSidebar"] .stButton > button {
    background: white !important;
    color: #1F3864 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    padding: 10px !important;
    font-size: 14px !important;
    margin-top: 6px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #E8F0FE !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
}

/* ── Predict button in main area ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1F3864, #2E74B5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    box-shadow: 0 2px 8px rgba(31,56,100,0.3);
}

/* ── Analytics chart backgrounds ── */
.chart-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 1px 5px rgba(0,0,0,0.07);
}

/* ── Footer caption ── */
.footer-bar {
    margin-top: 20px;
    padding: 12px 0 4px 0;
    border-top: 1px solid #dde2ec;
    text-align: center;
    font-size: 12px;
    color: #888;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND DATA
# ============================================================
@st.cache_resource
def load_model():
    model  = joblib.load('random_forest.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model()

@st.cache_data
def load_data():
    try:
        return pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
    except FileNotFoundError:
        return None

raw_df = load_data()

# ============================================================
# ENCODING MAPS (must match training)
# ============================================================
department_map      = {'Sales': 2, 'Research & Development': 1, 'Human Resources': 0}
job_role_map        = {
    'Sales Executive': 7, 'Research Scientist': 6, 'Laboratory Technician': 2,
    'Manufacturing Director': 4, 'Healthcare Representative': 0, 'Manager': 3,
    'Sales Representative': 8, 'Research Director': 5, 'Human Resources': 1
}
business_travel_map = {'Non-Travel': 0, 'Travel_Rarely': 2, 'Travel_Frequently': 1}
gender_map          = {'Female': 0, 'Male': 1}
marital_status_map  = {'Divorced': 0, 'Married': 1, 'Single': 2}
education_field_map = {
    'Life Sciences': 1, 'Medical': 3, 'Marketing': 2,
    'Technical Degree': 4, 'Human Resources': 0, 'Other': 5
}

# ============================================================
# HEADER — custom name logo
# ============================================================
st.markdown("""
<div class="name-logo">
    <div class="logo-circle">SJ</div>
    <div class="logo-text-block">
        <div class="logo-title">Employee Attrition Prediction Dashboard</div>
        <div class="logo-sub">
            Sheryl Ann Joseph &amp; Chaithrali Parabh &nbsp;·&nbsp;
            PGDM – Business Analytics &nbsp;·&nbsp;
            DCyber TechLab Pvt Ltd &nbsp;·&nbsp;
            Aditya School of Business Management, 2025–27
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 16px 0;'>
        <div style='font-size:28px;'>👤</div>
        <div style='font-size:15px; font-weight:700; color:white;
                    margin-top:4px;'>Employee Profile</div>
        <div style='font-size:11px; color:#BDD7EE; margin-top:2px;'>
            Adjust values to generate a prediction
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── Optional employee lookup ──
    selected_emp = None
    if raw_df is not None and 'EmployeeNumber' in raw_df.columns:
        emp_options = ['-- Manual Entry --'] + raw_df['EmployeeNumber'].astype(str).tolist()
        emp_choice  = st.selectbox('🔍 Load Existing Employee', emp_options)
        if emp_choice != '-- Manual Entry --':
            selected_emp = raw_df[raw_df['EmployeeNumber'] == int(emp_choice)].iloc[0]
        st.markdown("---")

    def default_val(col, fallback):
        if selected_emp is not None and col in selected_emp:
            return selected_emp[col]
        return fallback

    # ── Inputs ──
    st.markdown("<div class='section-header' style='color:white;border-color:white'>📌 Core Details</div>",
                unsafe_allow_html=True)

    age            = st.slider('Age', 18, 60, int(default_val('Age', 35)))
    monthly_income = st.number_input('Monthly Income (₹)', 1000, 200000,
                                      int(default_val('MonthlyIncome', 5000)), step=500)
    overtime_default = default_val('OverTime', 'No')
    overtime = st.selectbox('OverTime',['No','Yes'],
                             index=['No','Yes'].index(overtime_default)
                             if overtime_default in ['No','Yes'] else 0)
    job_level = st.selectbox('Job Level', [1,2,3,4,5],
                              index=int(default_val('JobLevel',1))-1)

    st.markdown("---")
    st.markdown("<div class='section-header' style='color:white;border-color:white'>😊 Satisfaction Scores</div>",
                unsafe_allow_html=True)

    job_sat  = st.slider('Job Satisfaction (1=Low, 4=High)', 1, 4,
                          int(default_val('JobSatisfaction', 3)))
    env_sat  = st.slider('Environment Satisfaction', 1, 4,
                          int(default_val('EnvironmentSatisfaction', 3)))
    work_life = st.slider('Work-Life Balance', 1, 4,
                           int(default_val('WorkLifeBalance', 3)))

    st.markdown("---")
    st.markdown("<div class='section-header' style='color:white;border-color:white'>🏢 Role & Tenure</div>",
                unsafe_allow_html=True)

    dept_default = default_val('Department', 'Sales')
    department   = st.selectbox('Department', list(department_map.keys()),
                                 index=list(department_map.keys()).index(dept_default)
                                 if dept_default in department_map else 0)
    role_default = default_val('JobRole', 'Sales Executive')
    job_role     = st.selectbox('Job Role', list(job_role_map.keys()),
                                 index=list(job_role_map.keys()).index(role_default)
                                 if role_default in job_role_map else 0)
    years_company  = st.slider('Years at Company', 0, 40,
                                int(default_val('YearsAtCompany', 5)))
    years_in_role  = st.slider('Years in Current Role', 0, 18,
                                int(default_val('YearsInCurrentRole', 2)))
    num_companies  = st.slider('Number of Companies Worked', 0, 9,
                                int(default_val('NumCompaniesWorked', 2)))
    distance       = st.slider('Distance from Home (km)', 1, 30,
                                int(default_val('DistanceFromHome', 5)))

    st.markdown("---")
    predict_clicked = st.button('🔍  Predict Attrition Risk', use_container_width=True)

# ============================================================
# ENCODE & BUILD INPUT
# ============================================================
overtime_enc   = 1 if overtime == 'Yes' else 0
department_enc = department_map[department]
job_role_enc   = job_role_map[job_role]

input_data = pd.DataFrame([{
    'Age':                      age,
    'BusinessTravel':           business_travel_map.get(default_val('BusinessTravel','Travel_Rarely'), 2),
    'DailyRate':                int(default_val('DailyRate', 800)),
    'Department':               department_enc,
    'DistanceFromHome':         distance,
    'Education':                int(default_val('Education', 3)),
    'EducationField':           education_field_map.get(default_val('EducationField','Life Sciences'), 1),
    'EmployeeCount':            1,
    'EnvironmentSatisfaction':  env_sat,
    'Gender':                   gender_map.get(default_val('Gender','Male'), 1),
    'HourlyRate':               int(default_val('HourlyRate', 65)),
    'JobInvolvement':           int(default_val('JobInvolvement', 3)),
    'JobLevel':                 job_level,
    'JobRole':                  job_role_enc,
    'JobSatisfaction':          job_sat,
    'MaritalStatus':            marital_status_map.get(default_val('MaritalStatus','Married'), 1),
    'MonthlyIncome':            monthly_income,
    'MonthlyRate':              int(default_val('MonthlyRate', 14000)),
    'NumCompaniesWorked':       num_companies,
    'OverTime':                 overtime_enc,
    'PercentSalaryHike':        int(default_val('PercentSalaryHike', 15)),
    'PerformanceRating':        int(default_val('PerformanceRating', 3)),
    'RelationshipSatisfaction': int(default_val('RelationshipSatisfaction', 3)),
    'StockOptionLevel':         int(default_val('StockOptionLevel', 1)),
    'TotalWorkingYears':        int(default_val('TotalWorkingYears', max(age-22, 1))),
    'TrainingTimesLastYear':    int(default_val('TrainingTimesLastYear', 3)),
    'WorkLifeBalance':          work_life,
    'YearsAtCompany':           years_company,
    'YearsInCurrentRole':       years_in_role,
    'YearsSinceLastPromotion':  int(default_val('YearsSinceLastPromotion', 1)),
    'YearsWithCurrManager':     int(default_val('YearsWithCurrManager', 3)),
}])

# ============================================================
# PREDICTION + SUMMARY
# ============================================================
col_pred, col_summary = st.columns([1.1, 1], gap='large')

with col_pred:
    st.markdown("<div class='section-header'>🧠 Attrition Prediction</div>",
                unsafe_allow_html=True)

    if predict_clicked:
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class='card-high'>
                <h3>⚠ HIGH RISK</h3>
                <h2>{probability:.1f}% probability of leaving</h2>
                <p>This employee shows multiple attrition risk signals.<br>
                   Recommend HR review and retention discussion within 30 days.</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(probability / 100)
        else:
            st.markdown(f"""
            <div class='card-low'>
                <h3>✅ LOW RISK</h3>
                <h2>Only {probability:.1f}% probability of leaving</h2>
                <p>This employee appears stable and engaged.<br>
                   Continue standard engagement monitoring.</p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(probability / 100)
    else:
        st.info('👈  Adjust the employee profile in the sidebar, then click **Predict Attrition Risk**.')

with col_summary:
    st.markdown("<div class='section-header'>📋 Input Summary</div>",
                unsafe_allow_html=True)

    rows = [
        ("Age",                    age),
        ("Monthly Income",         f"₹{monthly_income:,}"),
        ("OverTime",               overtime),
        ("Department",             department),
        ("Job Role",               job_role),
        ("Job Level",              job_level),
        ("Job Satisfaction",       f"{job_sat} / 4"),
        ("Environment Satisfaction", f"{env_sat} / 4"),
        ("Work-Life Balance",      f"{work_life} / 4"),
        ("Years at Company",       years_company),
        ("Years in Current Role",  years_in_role),
        ("Num. Companies Worked",  num_companies),
        ("Distance from Home",     f"{distance} km"),
    ]

    table_html = "<table class='summary-table'><tr><th>Attribute</th><th>Value</th></tr>"
    for attr, val in rows:
        table_html += f"<tr><td>{attr}</td><td><b>{val}</b></td></tr>"
    table_html += "</table>"
    st.markdown(table_html, unsafe_allow_html=True)

# ============================================================
# ANALYTICS SECTION
# ============================================================
st.divider()
st.markdown("<div class='section-header'>📊 Attrition Analytics — Full Dataset Overview</div>",
            unsafe_allow_html=True)

if raw_df is not None:
    chart1, chart2 = st.columns(2, gap='large')

    with chart1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#FAFBFC')
        dept_att = (raw_df.groupby('Department')['Attrition']
                    .value_counts(normalize=True).unstack() * 100)
        bars = dept_att['Yes'].sort_values().plot(
            kind='barh', ax=ax, color='#C0392B', edgecolor='white', linewidth=0.5
        )
        ax.set_title('Attrition Rate by Department (%)',
                     fontweight='bold', fontsize=12, color='#1F3864', pad=10)
        ax.set_xlabel('Attrition Rate (%)', fontsize=10, color='#555')
        ax.tick_params(colors='#555')
        ax.spines[['top','right']].set_visible(False)
        for i, v in enumerate(dept_att['Yes'].sort_values()):
            ax.text(v + 0.3, i, f'{v:.1f}%', va='center',
                    fontsize=10, color='#333', fontweight='600')
        plt.tight_layout()
        st.pyplot(fig)

    with chart2:
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        fig2.patch.set_facecolor('white')
        ax2.set_facecolor('#FAFBFC')
        sns.countplot(x='OverTime', hue='Attrition', data=raw_df,
                      palette=['#2E74B5','#C0392B'], ax=ax2,
                      edgecolor='white', linewidth=0.5)
        ax2.set_title('Overtime vs Attrition',
                      fontweight='bold', fontsize=12, color='#1F3864', pad=10)
        ax2.set_xlabel('OverTime Status', fontsize=10, color='#555')
        ax2.set_ylabel('Number of Employees', fontsize=10, color='#555')
        ax2.tick_params(colors='#555')
        ax2.spines[['top','right']].set_visible(False)
        ax2.legend(title='Attrition', title_fontsize=9, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)

else:
    st.info('Dataset not found. Upload WA_Fn-UseC_-HR-Employee-Attrition.csv to your repo.')

# ============================================================
# MODEL METRICS FOOTER
# ============================================================
st.divider()
st.markdown("<div class='section-header'>⚙ Model Performance — Random Forest</div>",
            unsafe_allow_html=True)

st.markdown("""
<div class='metric-row'>
    <div class='metric-box'>
        <div class='m-label'>Algorithm</div>
        <div class='m-value'>🌲 RF</div>
        <div class='m-sub'>200 trees</div>
    </div>
    <div class='metric-box'>
        <div class='m-label'>Accuracy</div>
        <div class='m-value'>84.01%</div>
        <div class='m-sub'>Test set</div>
    </div>
    <div class='metric-box'>
        <div class='m-label'>AUC-ROC</div>
        <div class='m-value'>0.822</div>
        <div class='m-sub'>Best of 3 models</div>
    </div>
    <div class='metric-box'>
        <div class='m-label'>CV F1 Score</div>
        <div class='m-value'>0.908</div>
        <div class='m-sub'>5-fold mean</div>
    </div>
    <div class='metric-box'>
        <div class='m-label'>Training Data</div>
        <div class='m-value'>1,470</div>
        <div class='m-sub'>IBM HR records</div>
    </div>
    <div class='metric-box'>
        <div class='m-label'>Class Balance</div>
        <div class='m-value'>SMOTE</div>
        <div class='m-sub'>986 / 986</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class='footer-bar'>
    🏢 &nbsp; Built by <b>Sheryl Ann Joseph</b> &amp; <b>Chaithrali Parabh</b> &nbsp;|&nbsp;
    Summer Internship Project &nbsp;|&nbsp;
    DCyber TechLab Pvt Ltd, Mumbai &nbsp;|&nbsp;
    Aditya School of Business Management — PGDM Business Analytics 2025–27
</div>
""", unsafe_allow_html=True)
