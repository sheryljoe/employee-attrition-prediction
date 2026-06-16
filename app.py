# app.py  –  Employee Attrition Prediction Dashboard  (v2 – Enhanced UI)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title='Employee Attrition Predictor',
    page_icon='🏢',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================================
# MASTER CSS
# ============================================================
st.markdown("""
<style>
/* ── Reset & base ── */
#MainMenu, footer, header {visibility: hidden;}
* {box-sizing: border-box;}
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
}

/* ── Page background – subtle blue-grey gradient ── */
.stApp {
    background: linear-gradient(145deg, #EEF2F7 0%, #E8EDF5 40%, #EDF1F8 100%);
    min-height: 100vh;
}
.main .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ══════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1E3D 0%, #1A3A6B 55%, #1F4A8A 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
    box-shadow: 4px 0 20px rgba(0,0,0,0.18);
}
[data-testid="stSidebar"] > div:first-child {padding-top: 0;}

/* All sidebar text */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #E0EAFF !important;
}
/* Input labels – bright white, readable */
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label {
    color: #FFFFFF !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}
/* Slider value text */
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
[data-testid="stSidebar"] .stSlider .st-emotion-cache-1wq3kxq {
    color: #7DD3FC !important;
    font-weight: 700;
}
/* Selectbox & number input fields */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.12);
    margin: 14px 0;
}

/* Sidebar predict button */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #3B82F6, #1D4ED8) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 0 !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.45) !important;
    letter-spacing: 0.04em;
    transition: all .2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #60A5FA, #2563EB) !important;
    box-shadow: 0 6px 20px rgba(59,130,246,0.55) !important;
    transform: translateY(-1px);
}

/* ══════════════════════════════════════════
   HEADER BANNER
══════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg, #0D1E3D 0%, #1A3A6B 50%, #1E4D8C 100%);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 28px;
    box-shadow: 0 8px 32px rgba(13,30,61,0.28);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(59,130,246,0.12);
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}

/* Name logo circle */
.hero-logo {
    flex-shrink: 0;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
}

.hero-text {z-index: 1;}
.hero-title {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin: 0 0 6px 0;
}
.hero-subtitle {
    font-size: 13.5px;
    color: #93B4DC;
    margin: 0;
    line-height: 1.6;
}
.hero-subtitle b {color: #BDD7EE;}
.hero-badge {
    margin-left: auto;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 6px;
    z-index: 1;
    flex-shrink: 0;
}
.hero-badge .badge-pill {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 11.5px;
    color: #BDD7EE;
    font-weight: 600;
    white-space: nowrap;
}

/* ══════════════════════════════════════════
   SECTION HEADERS
══════════════════════════════════════════ */
.sec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 14px 0;
}
.sec-header .icon-box {
    width: 36px; height: 36px;
    border-radius: 9px;
    background: linear-gradient(135deg, #1A3A6B, #3B82F6);
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
    box-shadow: 0 3px 10px rgba(59,130,246,0.30);
}
.sec-header .sec-title {
    font-size: 17px;
    font-weight: 700;
    color: #0D1E3D;
    letter-spacing: -0.2px;
}

/* ══════════════════════════════════════════
   PREDICTION CARDS
══════════════════════════════════════════ */
.pred-card-high {
    background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
    border: 1.5px solid #FECACA;
    border-left: 6px solid #DC2626;
    border-radius: 16px;
    padding: 24px 26px;
    box-shadow: 0 4px 20px rgba(220,38,38,0.12);
}
.pred-card-high .pc-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #DC2626; color: white;
    border-radius: 20px; padding: 4px 14px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
    margin-bottom: 12px;
}
.pred-card-high .pc-prob {
    font-size: 48px; font-weight: 900;
    color: #B91C1C; line-height: 1; margin: 0;
}
.pred-card-high .pc-label {
    font-size: 14px; color: #7F1D1D;
    margin: 6px 0 12px 0; font-weight: 500;
}
.pred-card-high .pc-desc {
    font-size: 13px; color: #991B1B;
    background: rgba(220,38,38,0.08);
    border-radius: 8px; padding: 10px 12px; margin: 0;
}

.pred-card-low {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7);
    border: 1.5px solid #BBF7D0;
    border-left: 6px solid #16A34A;
    border-radius: 16px;
    padding: 24px 26px;
    box-shadow: 0 4px 20px rgba(22,163,74,0.12);
}
.pred-card-low .pc-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #16A34A; color: white;
    border-radius: 20px; padding: 4px 14px;
    font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
    margin-bottom: 12px;
}
.pred-card-low .pc-prob {
    font-size: 48px; font-weight: 900;
    color: #15803D; line-height: 1; margin: 0;
}
.pred-card-low .pc-label {
    font-size: 14px; color: #14532D;
    margin: 6px 0 12px 0; font-weight: 500;
}
.pred-card-low .pc-desc {
    font-size: 13px; color: #166534;
    background: rgba(22,163,74,0.08);
    border-radius: 8px; padding: 10px 12px; margin: 0;
}

.pred-idle {
    background: linear-gradient(135deg, #F8FAFF, #EEF4FF);
    border: 1.5px solid #C7D8F0;
    border-radius: 16px;
    padding: 40px 26px;
    text-align: center;
    color: #4A6FA5;
}
.pred-idle .idle-icon {font-size: 48px; margin-bottom: 14px;}
.pred-idle p {font-size: 14px; color: #5A7BB5; margin: 0;}

/* ══════════════════════════════════════════
   SUMMARY TABLE
══════════════════════════════════════════ */
.sum-table {
    width: 100%; border-collapse: collapse;
    font-size: 13px; border-radius: 12px;
    overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.sum-table th {
    background: linear-gradient(135deg, #0D1E3D, #1A3A6B);
    color: white; padding: 11px 16px;
    text-align: left; font-weight: 600;
    font-size: 12px; text-transform: uppercase;
    letter-spacing: 0.06em;
}
.sum-table td {
    padding: 9px 16px;
    border-bottom: 1px solid #E8EDF5;
    color: #1E293B;
}
.sum-table td:last-child {
    font-weight: 600; color: #0D1E3D;
}
.sum-table tr:nth-child(even) td {background: #F4F7FC;}
.sum-table tr:last-child td {border-bottom: none;}
.sum-table tr:hover td {background: #EBF2FF; transition: background .15s;}

/* ══════════════════════════════════════════
   METRIC CARDS (footer)
══════════════════════════════════════════ */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 14px;
    margin-top: 8px;
}
.metric-card {
    background: white;
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 4px solid #3B82F6;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.metric-card.green  {border-top-color: #16A34A;}
.metric-card.purple {border-top-color: #7C3AED;}
.metric-card.amber  {border-top-color: #D97706;}
.metric-card.teal   {border-top-color: #0D9488;}
.metric-card.rose   {border-top-color: #E11D48;}
.m-icon  {font-size: 22px; margin-bottom: 6px;}
.m-lbl   {font-size: 10.5px; color: #6B7280; font-weight: 700;
           text-transform: uppercase; letter-spacing: .07em;}
.m-val   {font-size: 22px; font-weight: 800; color: #0D1E3D;
           margin: 4px 0 2px;}
.m-sub   {font-size: 10.5px; color: #9CA3AF;}

/* ══════════════════════════════════════════
   CHART CONTAINER
══════════════════════════════════════════ */
.chart-wrap {
    background: white;
    border-radius: 16px;
    padding: 22px 22px 16px 22px;
    box-shadow: 0 2px 14px rgba(0,0,0,0.07);
    border: 1px solid #E8EDF5;
}

/* ══════════════════════════════════════════
   SIDEBAR SECTION LABELS
══════════════════════════════════════════ */
.sb-sec {
    font-size: 10px;
    font-weight: 800;
    color: #7DD3FC;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 6px 0 4px 2px;
    margin-top: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.10);
    margin-bottom: 10px;
}

/* ══════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════ */
.styled-div {
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%, #C7D8F0 20%, #8BAFD4 50%, #C7D8F0 80%, transparent 100%);
    border: none;
    margin: 28px 0;
    border-radius: 2px;
}

/* ══════════════════════════════════════════
   FOOTER
══════════════════════════════════════════ */
.footer-strip {
    margin-top: 32px;
    padding: 14px 0 6px 0;
    border-top: 1px solid #D1DCF0;
    text-align: center;
    font-size: 12px;
    color: #6B7280;
    line-height: 1.8;
}
.footer-strip b {color: #0D1E3D;}

/* ══════════════════════════════════════════
   PROGRESS BAR OVERRIDE
══════════════════════════════════════════ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3B82F6, #1D4ED8) !important;
    border-radius: 10px;
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
# ENCODING MAPS
# ============================================================
department_map = {'Sales': 2, 'Research & Development': 1, 'Human Resources': 0}
job_role_map = {
    'Sales Executive': 7, 'Research Scientist': 6, 'Laboratory Technician': 2,
    'Manufacturing Director': 4, 'Healthcare Representative': 0, 'Manager': 3,
    'Sales Representative': 8, 'Research Director': 5, 'Human Resources': 1
}
business_travel_map  = {'Non-Travel': 0, 'Travel_Rarely': 2, 'Travel_Frequently': 1}
gender_map           = {'Female': 0, 'Male': 1}
marital_status_map   = {'Divorced': 0, 'Married': 1, 'Single': 2}
education_field_map  = {
    'Life Sciences': 1, 'Medical': 3, 'Marketing': 2,
    'Technical Degree': 4, 'Human Resources': 0, 'Other': 5
}

# ============================================================
# HERO HEADER
# ============================================================
st.markdown("""
<div class="hero-banner">
    <div class="hero-logo">
        <!-- Company-style SVG logo for Sheryl Joseph -->
        <svg width="96" height="96" viewBox="0 0 96 96" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Outer ring -->
            <circle cx="48" cy="48" r="46" fill="url(#outerGrad)" opacity="0.15"/>
            <!-- Main rounded square background -->
            <rect x="6" y="6" width="84" height="84" rx="22" fill="url(#mainGrad)"/>
            <!-- Inner glow ring -->
            <rect x="6" y="6" width="84" height="84" rx="22"
                  fill="none" stroke="url(#ringGrad)" stroke-width="1.5"/>
            <!-- Top accent bar -->
            <rect x="18" y="16" width="60" height="5" rx="2.5" fill="white" opacity="0.18"/>
            <!-- Letter S — bold custom path -->
            <text x="48" y="52" text-anchor="middle" dominant-baseline="middle"
                  font-family="Segoe UI, Arial Black, sans-serif"
                  font-size="38" font-weight="900" fill="white"
                  letter-spacing="-1">S</text>
            <!-- Small "J" superscript pill -->
            <rect x="56" y="22" width="20" height="13" rx="6.5" fill="white" opacity="0.22"/>
            <text x="66" y="31" text-anchor="middle" dominant-baseline="middle"
                  font-family="Segoe UI, Arial, sans-serif"
                  font-size="9" font-weight="800" fill="white">AJ</text>
            <!-- Bottom tagline bar -->
            <rect x="22" y="70" width="52" height="3.5" rx="1.75"
                  fill="white" opacity="0.20"/>
            <!-- Defs -->
            <defs>
                <linearGradient id="mainGrad" x1="0" y1="0" x2="96" y2="96" gradientUnits="userSpaceOnUse">
                    <stop offset="0%"   stop-color="#1E40AF"/>
                    <stop offset="50%"  stop-color="#2563EB"/>
                    <stop offset="100%" stop-color="#1D4ED8"/>
                </linearGradient>
                <linearGradient id="outerGrad" x1="0" y1="0" x2="96" y2="96" gradientUnits="userSpaceOnUse">
                    <stop offset="0%"   stop-color="#60A5FA"/>
                    <stop offset="100%" stop-color="#3B82F6"/>
                </linearGradient>
                <linearGradient id="ringGrad" x1="0" y1="0" x2="96" y2="96" gradientUnits="userSpaceOnUse">
                    <stop offset="0%"   stop-color="rgba(255,255,255,0.40)"/>
                    <stop offset="100%" stop-color="rgba(255,255,255,0.05)"/>
                </linearGradient>
            </defs>
        </svg>
    </div>
    <div class="hero-text">
        <div class="hero-title">Employee Attrition Prediction Dashboard</div>
        <div class="hero-subtitle">
            <b>Sheryl Ann Joseph</b> &nbsp;·&nbsp;
            DCyber TechLab Pvt Ltd, Mumbai &nbsp;·&nbsp;
            Aditya School of Business Management
        </div>
    </div>
    <div class="hero-badge">
        <span class="badge-pill">🤖 Random Forest Model</span>
        <span class="badge-pill">📊 IBM HR Analytics Dataset</span>
        <span class="badge-pill">🚀 Live Deployment</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR – Dashboard Filters
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="background:rgba(255,255,255,0.06); border-radius:12px;
                padding:16px 14px 12px 14px; margin-bottom:10px; text-align:center;">
        <div style="font-size:32px; margin-bottom:6px;">⚙️</div>
        <div style="font-size:16px; font-weight:800; color:#FFFFFF;
                    letter-spacing:0.03em;">Dashboard Filters</div>
        <div style="font-size:11px; color:#93B4DC; margin-top:3px;">
            Configure employee profile to generate prediction
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Optional employee lookup ──
    selected_emp = None
    if raw_df is not None and 'EmployeeNumber' in raw_df.columns:
        st.markdown("<div class='sb-sec'>🔍 Employee Lookup</div>", unsafe_allow_html=True)
        emp_options = ['— Manual Entry —'] + raw_df['EmployeeNumber'].astype(str).tolist()
        emp_choice  = st.selectbox('Load Existing Employee', emp_options, label_visibility='collapsed')
        if emp_choice != '— Manual Entry —':
            selected_emp = raw_df[raw_df['EmployeeNumber'] == int(emp_choice)].iloc[0]
        st.markdown("<hr>", unsafe_allow_html=True)

    def dv(col, fallback):
        if selected_emp is not None and col in selected_emp:
            return selected_emp[col]
        return fallback

    # ── Section 1: Core Details ──
    st.markdown("<div class='sb-sec'>📌 Core Details</div>", unsafe_allow_html=True)
    age            = st.slider('Age', 18, 60, int(dv('Age', 35)))
    monthly_income = st.number_input('Monthly Income (₹)', 1000, 200000,
                                      int(dv('MonthlyIncome', 5000)), step=500)
    ot_default = dv('OverTime', 'No')
    overtime = st.selectbox('OverTime Status',['No','Yes'],
                             index=['No','Yes'].index(ot_default)
                             if ot_default in ['No','Yes'] else 0)
    job_level = st.selectbox('Job Level', [1,2,3,4,5],
                              index=int(dv('JobLevel',1))-1)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 2: Satisfaction ──
    st.markdown("<div class='sb-sec'>😊 Satisfaction Scores</div>", unsafe_allow_html=True)
    job_sat   = st.slider('Job Satisfaction',          1, 4, int(dv('JobSatisfaction', 3)),
                           help="1 = Low, 4 = High")
    env_sat   = st.slider('Environment Satisfaction',  1, 4, int(dv('EnvironmentSatisfaction', 3)),
                           help="1 = Low, 4 = High")
    work_life = st.slider('Work-Life Balance',         1, 4, int(dv('WorkLifeBalance', 3)),
                           help="1 = Low, 4 = High")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Section 3: Role & Tenure ──
    st.markdown("<div class='sb-sec'>🏢 Role & Tenure</div>", unsafe_allow_html=True)
    dept_default = dv('Department', 'Sales')
    department   = st.selectbox('Department', list(department_map.keys()),
                                 index=list(department_map.keys()).index(dept_default)
                                 if dept_default in department_map else 0)
    role_default = dv('JobRole', 'Sales Executive')
    job_role     = st.selectbox('Job Role', list(job_role_map.keys()),
                                 index=list(job_role_map.keys()).index(role_default)
                                 if role_default in job_role_map else 0)
    years_company  = st.slider('Years at Company',      0, 40, int(dv('YearsAtCompany', 5)))
    years_in_role  = st.slider('Years in Current Role', 0, 18, int(dv('YearsInCurrentRole', 2)))
    num_companies  = st.slider('No. of Companies Worked', 0, 9, int(dv('NumCompaniesWorked', 2)))
    distance       = st.slider('Distance from Home (km)', 1, 30, int(dv('DistanceFromHome', 5)))

    st.markdown("<hr>", unsafe_allow_html=True)
    predict_clicked = st.button('🔍  Generate Prediction', use_container_width=True)


# ============================================================
# BUILD INPUT ARRAY
# ============================================================
overtime_enc   = 1 if overtime == 'Yes' else 0
department_enc = department_map[department]
job_role_enc   = job_role_map[job_role]

input_data = pd.DataFrame([{
    'Age':                      age,
    'BusinessTravel':           business_travel_map.get(dv('BusinessTravel','Travel_Rarely'), 2),
    'DailyRate':                int(dv('DailyRate', 800)),
    'Department':               department_enc,
    'DistanceFromHome':         distance,
    'Education':                int(dv('Education', 3)),
    'EducationField':           education_field_map.get(dv('EducationField','Life Sciences'), 1),
    'EmployeeCount':            1,
    'EnvironmentSatisfaction':  env_sat,
    'Gender':                   gender_map.get(dv('Gender','Male'), 1),
    'HourlyRate':               int(dv('HourlyRate', 65)),
    'JobInvolvement':           int(dv('JobInvolvement', 3)),
    'JobLevel':                 job_level,
    'JobRole':                  job_role_enc,
    'JobSatisfaction':          job_sat,
    'MaritalStatus':            marital_status_map.get(dv('MaritalStatus','Married'), 1),
    'MonthlyIncome':            monthly_income,
    'MonthlyRate':              int(dv('MonthlyRate', 14000)),
    'NumCompaniesWorked':       num_companies,
    'OverTime':                 overtime_enc,
    'PercentSalaryHike':        int(dv('PercentSalaryHike', 15)),
    'PerformanceRating':        int(dv('PerformanceRating', 3)),
    'RelationshipSatisfaction': int(dv('RelationshipSatisfaction', 3)),
    'StockOptionLevel':         int(dv('StockOptionLevel', 1)),
    'TotalWorkingYears':        int(dv('TotalWorkingYears', max(age-22, 1))),
    'TrainingTimesLastYear':    int(dv('TrainingTimesLastYear', 3)),
    'WorkLifeBalance':          work_life,
    'YearsAtCompany':           years_company,
    'YearsInCurrentRole':       years_in_role,
    'YearsSinceLastPromotion':  int(dv('YearsSinceLastPromotion', 1)),
    'YearsWithCurrManager':     int(dv('YearsWithCurrManager', 3)),
}])


# ============================================================
# PREDICTION + SUMMARY ROW
# ============================================================
st.markdown("""
<div class="sec-header">
    <div class="icon-box">🧠</div>
    <div class="sec-title">Attrition Prediction &amp; Employee Profile</div>
</div>
""", unsafe_allow_html=True)

col_pred, col_gap, col_sum = st.columns([1.05, 0.06, 1], gap='small')

with col_pred:
    if predict_clicked:
        input_scaled = scaler.transform(input_data)
        prediction   = model.predict(input_scaled)[0]
        probability  = model.predict_proba(input_scaled)[0][1] * 100

        if prediction == 1:
            st.markdown(f"""
            <div class='pred-card-high'>
                <div class='pc-badge'>⚠ HIGH ATTRITION RISK</div>
                <div class='pc-prob'>{probability:.1f}%</div>
                <div class='pc-label'>probability this employee will leave</div>
                <div class='pc-desc'>
                    This employee profile matches multiple high-risk attrition signals.<br>
                    Recommend a targeted HR retention discussion within <b>30 days</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='pred-card-low'>
                <div class='pc-badge'>✅ LOW ATTRITION RISK</div>
                <div class='pc-prob'>{probability:.1f}%</div>
                <div class='pc-label'>probability this employee will leave</div>
                <div class='pc-desc'>
                    This employee appears stable and engaged with the organisation.<br>
                    Continue standard engagement and monitoring practices.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.progress(probability / 100)
        st.caption(f"Risk Score: **{probability:.1f}%** &nbsp;|&nbsp; "
                   f"Model confidence based on Random Forest (AUC = 0.822)")
    else:
        st.markdown("""
        <div class='pred-idle'>
            <div class='idle-icon'>🔍</div>
            <p><b>No prediction yet</b><br><br>
            Configure the employee profile using the<br>
            <b>Dashboard Filters</b> on the left, then click<br>
            <b>Generate Prediction</b> to see the result.</p>
        </div>
        """, unsafe_allow_html=True)

with col_sum:
    rows = [
        ("Age",                     f"{age} yrs"),
        ("Monthly Income",          f"₹{monthly_income:,}"),
        ("OverTime",                overtime),
        ("Department",              department),
        ("Job Role",                job_role),
        ("Job Level",               f"Level {job_level}"),
        ("Job Satisfaction",        f"{job_sat} / 4"),
        ("Environment Satisfaction",f"{env_sat} / 4"),
        ("Work-Life Balance",       f"{work_life} / 4"),
        ("Years at Company",        f"{years_company} yrs"),
        ("Years in Current Role",   f"{years_in_role} yrs"),
        ("No. Companies Worked",    num_companies),
        ("Distance from Home",      f"{distance} km"),
    ]
    tbl = "<table class='sum-table'><tr><th>Attribute</th><th>Value</th></tr>"
    for a, v in rows:
        tbl += f"<tr><td>{a}</td><td>{v}</td></tr>"
    tbl += "</table>"
    st.markdown(tbl, unsafe_allow_html=True)


# ============================================================
# ANALYTICS SECTION
# ============================================================
st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
    <div class="icon-box">📊</div>
    <div class="sec-title">Attrition Analytics — IBM HR Dataset Overview (1,470 Employees)</div>
</div>
""", unsafe_allow_html=True)

# ── Matplotlib style ──
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'axes.grid.axis':    'x',
    'grid.color':        '#E8EDF5',
    'grid.linewidth':    0.8,
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
})

PALETTE_MAIN  = ['#1A3A6B', '#3B82F6', '#60A5FA', '#93C5FD']
COLOR_HIGH    = '#DC2626'
COLOR_STAY    = '#1A3A6B'
COLOR_LEAVE   = '#DC2626'

if raw_df is not None:
    ch1, ch2 = st.columns(2, gap='large')

    # ── Chart 1: Attrition by Department ──
    with ch1:
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        dept_att = (raw_df.groupby('Department')['Attrition']
                    .value_counts(normalize=True).unstack() * 100)
        vals   = dept_att['Yes'].sort_values()
        colors = ['#60A5FA' if v < 18 else '#DC2626' for v in vals]
        bars = ax.barh(vals.index, vals.values, color=colors,
                       edgecolor='white', linewidth=1, height=0.55)
        for bar, val in zip(bars, vals.values):
            ax.text(val + 0.4, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', ha='left',
                    fontsize=11, fontweight='700', color='#1E293B')
        ax.set_title('Attrition Rate by Department',
                     fontsize=13, fontweight='800', color='#0D1E3D',
                     pad=12, loc='left')
        ax.set_xlabel('Attrition Rate (%)', fontsize=10, color='#6B7280')
        ax.tick_params(axis='y', labelsize=11, colors='#374151')
        ax.tick_params(axis='x', labelsize=9,  colors='#6B7280')
        ax.set_xlim(0, vals.max() + 8)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Chart 2: Overtime vs Attrition ──
    with ch2:
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(6.5, 3.8))
        ot_data = raw_df.groupby(['OverTime','Attrition']).size().unstack(fill_value=0)
        ot_pct  = ot_data.div(ot_data.sum(axis=1), axis=0) * 100
        x = np.arange(len(ot_pct))
        w = 0.35
        b1 = ax2.bar(x - w/2, ot_pct['No'],  w, color=COLOR_STAY,
                     label='Stayed', edgecolor='white', linewidth=1)
        b2 = ax2.bar(x + w/2, ot_pct['Yes'], w, color=COLOR_LEAVE,
                     label='Left',   edgecolor='white', linewidth=1)
        for bar in [*b1, *b2]:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.8,
                     f'{h:.1f}%', ha='center', va='bottom',
                     fontsize=10, fontweight='700', color='#1E293B')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['No Overtime', 'Overtime'], fontsize=11, color='#374151')
        ax2.set_title('Overtime vs Attrition Rate',
                      fontsize=13, fontweight='800', color='#0D1E3D',
                      pad=12, loc='left')
        ax2.set_ylabel('Percentage (%)', fontsize=10, color='#6B7280')
        ax2.tick_params(axis='y', labelsize=9, colors='#6B7280')
        ax2.set_ylim(0, 105)
        legend = ax2.legend(fontsize=10, framealpha=0.85, edgecolor='#E8EDF5',
                             loc='upper right', ncol=2)
        legend.get_frame().set_linewidth(1)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Second row: Income distribution + Job Role attrition ──
    ch3, ch4 = st.columns(2, gap='large')

    with ch3:
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(6.5, 3.8))
        stayed = raw_df[raw_df['Attrition']=='No']['MonthlyIncome']
        left   = raw_df[raw_df['Attrition']=='Yes']['MonthlyIncome']
        ax3.hist(stayed, bins=25, alpha=0.72, color=COLOR_STAY,
                 label='Stayed', edgecolor='white', linewidth=0.5)
        ax3.hist(left,   bins=25, alpha=0.72, color=COLOR_LEAVE,
                 label='Left',   edgecolor='white', linewidth=0.5)
        ax3.axvline(stayed.mean(), color=COLOR_STAY,  lw=2, ls='--',
                    label=f'Avg Stayed ₹{stayed.mean():,.0f}')
        ax3.axvline(left.mean(),   color=COLOR_LEAVE, lw=2, ls='--',
                    label=f'Avg Left ₹{left.mean():,.0f}')
        ax3.set_title('Monthly Income Distribution by Attrition',
                      fontsize=13, fontweight='800', color='#0D1E3D',
                      pad=12, loc='left')
        ax3.set_xlabel('Monthly Income (₹)', fontsize=10, color='#6B7280')
        ax3.set_ylabel('No. of Employees',   fontsize=10, color='#6B7280')
        ax3.tick_params(labelsize=9, colors='#6B7280')
        ax3.legend(fontsize=9, framealpha=0.85, edgecolor='#E8EDF5')
        ax3.grid(axis='y')
        ax3.grid(axis='x', visible=False)
        plt.tight_layout(pad=1.5)
        st.pyplot(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with ch4:
        st.markdown("<div class='chart-wrap'>", unsafe_allow_html=True)
        fig4, ax4 = plt.subplots(figsize=(6.5, 3.8))
        role_att = (raw_df.groupby('JobRole')['Attrition']
                    .value_counts(normalize=True).unstack() * 100)['Yes'].sort_values()
        colors4  = ['#DC2626' if v >= role_att.mean() else '#60A5FA'
                    for v in role_att]
        hbars = ax4.barh(role_att.index, role_att.values,
                          color=colors4, edgecolor='white',
                          linewidth=0.8, height=0.6)
        ax4.axvline(role_att.mean(), color='#6B7280', lw=1.5, ls='--',
                    label=f'Average {role_att.mean():.1f}%', zorder=3)
        for bar, val in zip(hbars, role_att.values):
            ax4.text(val + 0.4, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', ha='left',
                     fontsize=9, fontweight='600', color='#1E293B')
        ax4.set_title('Attrition Rate by Job Role',
                      fontsize=13, fontweight='800', color='#0D1E3D',
                      pad=12, loc='left')
        ax4.set_xlabel('Attrition Rate (%)', fontsize=10, color='#6B7280')
        ax4.tick_params(axis='y', labelsize=9, colors='#374151')
        ax4.tick_params(axis='x', labelsize=9, colors='#6B7280')
        ax4.set_xlim(0, role_att.max() + 12)
        ax4.legend(fontsize=9, framealpha=0.85, edgecolor='#E8EDF5')
        plt.tight_layout(pad=1.5)
        st.pyplot(fig4, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info('Dataset not found. Upload **WA_Fn-UseC_-HR-Employee-Attrition.csv** to see analytics.')


# ============================================================
# MODEL METRICS FOOTER
# ============================================================
st.markdown("<div class='styled-div'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="sec-header">
    <div class="icon-box">⚙️</div>
    <div class="sec-title">Model Performance — Random Forest Classifier</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='metric-grid'>
    <div class='metric-card'>
        <div class='m-icon'>🌲</div>
        <div class='m-lbl'>Algorithm</div>
        <div class='m-val' style='font-size:17px;'>Random Forest</div>
        <div class='m-sub'>200 estimators</div>
    </div>
    <div class='metric-card green'>
        <div class='m-icon'>🎯</div>
        <div class='m-lbl'>Accuracy</div>
        <div class='m-val'>84.01%</div>
        <div class='m-sub'>Test set</div>
    </div>
    <div class='metric-card purple'>
        <div class='m-icon'>📈</div>
        <div class='m-lbl'>AUC-ROC</div>
        <div class='m-val'>0.822</div>
        <div class='m-sub'>Best of 3 models</div>
    </div>
    <div class='metric-card amber'>
        <div class='m-icon'>⚖️</div>
        <div class='m-lbl'>CV F1 Score</div>
        <div class='m-val'>0.908</div>
        <div class='m-sub'>5-fold mean</div>
    </div>
    <div class='metric-card teal'>
        <div class='m-icon'>🗃️</div>
        <div class='m-lbl'>Training Records</div>
        <div class='m-val'>1,470</div>
        <div class='m-sub'>IBM HR Dataset</div>
    </div>
    <div class='metric-card rose'>
        <div class='m-icon'>⚗️</div>
        <div class='m-lbl'>Class Balance</div>
        <div class='m-val'>SMOTE</div>
        <div class='m-sub'>986 / 986 balanced</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='footer-strip'>
    🏢 &nbsp;
    Built by <b>Sheryl Ann Joseph</b> &nbsp;·&nbsp;
    Summer Internship Project 2025–26 &nbsp;·&nbsp;
    DCyber TechLab Pvt Ltd, Mumbai &nbsp;·&nbsp;
    Aditya School of Business Management
</div>
""", unsafe_allow_html=True)
