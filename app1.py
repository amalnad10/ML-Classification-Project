import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Employee Attrition Intelligence System",
    page_icon="👨‍💼",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("best_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoders = joblib.load("label_encoders.joblib")
feature_names = joblib.load("feature_names.joblib")


# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.stApp{
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 100%
    );
}

.main{
    padding-top:20px;
}

.metric-card{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
    padding:12px;
    border-radius:10px;
    text-align:center;
    color:white;
    
}

.block-container{
    padding-top:2rem;
    padding-bottom:1rem;
}

section[data-testid="stSidebar"]{
    width:280px !important;
}

.big-title{
    text-align:center;
    color:white;
    font-size:2.3rem;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:1.1rem;
}

.stButton button{
    width:100%;
    height:60px;
    font-size:20px;
    font-weight:bold;
    border-radius:12px;
    background:#2563eb;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="big-title">👨‍💼 Employee Attrition Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">AI Powered Employee Retention Analytics</div>',
    unsafe_allow_html=True
)

# ---------------- HR DASHBOARD STATS ---------------- #

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>👥 Total Employees</h4>
        <h3>1470</h3>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h4>🚪 Employees Left</h4>
        <h3>237</h3>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>📈 Attrition Rate</h4>
        <h3>16.1%</h3>
    </div>
    """, unsafe_allow_html=True)



with st.sidebar:


    st.title("👤 Employee Information")

    st.markdown("---")

    age = st.slider(
        "Age",
        18,
        60,
        30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        1000,
        50000,
        5000
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    years_at_company = st.slider(
        "Years At Company",
        0,
        40,
        5
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Healthcare Representative",
            "Human Resources",
            "Laboratory Technician",
            "Manager",
            "Manufacturing Director",
            "Research Director",
            "Research Scientist",
            "Sales Executive",
            "Sales Representative"
        ]
    )

    job_satisfaction = st.slider(
        "Job Satisfaction",
        1,
        4,
        3
    )

    st.markdown("---")

    predict_btn = st.button(
        "🚀 Predict Attrition Risk",
        use_container_width=True
    )

    st.subheader("🎯 Objective")

    st.info("""
    Predict employee attrition risk
    and assist HR teams in employee
    retention decision-making.
    """)



if predict_btn:

    # Encode categorical values
    overtime_encoded = label_encoders["OverTime"].transform([overtime])[0]
    department_encoded = label_encoders["Department"].transform([department])[0]
    jobrole_encoded = label_encoders["JobRole"].transform([job_role])[0]

    # Create dataframe
    input_data = pd.DataFrame([{
        "Age": age,
        "MonthlyIncome": monthly_income,
        "OverTime": overtime_encoded,
        "YearsAtCompany": years_at_company,
        "Department": department_encoded,
        "JobRole": jobrole_encoded,
        "JobSatisfaction": job_satisfaction
    }])

    # Arrange columns correctly
    input_data = input_data[feature_names]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
       
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Risk Level
    if probability < 30:
        risk_text = "🟢 Low Risk"
    elif probability < 60:
        risk_text = "🟡 Medium Risk"
    else:
        risk_text = "🔴 High Risk"

    st.markdown("---")

# Prediction Result Card
    left, center, right = st.columns([1,2,1])

    with center:

        st.markdown(f"""
    <div class="metric-card">
        <h2>📊 Attrition Risk Analysis</h2>
        <h1>{probability:.1f}%</h1>
        <h3>{risk_text}</h3>
    </div>
    """, unsafe_allow_html=True)

        st.markdown("---")

    chart1, chart2 = st.columns(2)

    with chart1:

        st.subheader("🍩 Risk Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Attrition Risk", "Retention Probability"],
            "Value": [probability, 100 - probability]
        })

        fig1 = px.pie(
            chart_data,
            names="Category",
            values="Value",
            hole=0.6
        )

        fig1.update_layout(height=350)

        st.plotly_chart(fig1, use_container_width=True)

    with chart2:

        st.subheader("📌 Top Factors")

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": abs(model.coef_[0])
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        ).head(5)

        fig2 = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h"
        )

        fig2.update_layout(
            height=350,
            yaxis=dict(categoryorder="total ascending")
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- RECOMMENDATIONS ---------------- #

    if probability < 30:
    

       st.success("""
    ### ✅ Recommendations

    • Employee shows low attrition risk

    • Continue current engagement strategy

    • Maintain regular feedback sessions

    • Encourage skill development programs
    """)

    elif probability < 60:

        st.warning("""
    ### ⚠ Recommendations

    • Monitor employee satisfaction closely

    • Schedule career development discussions

    • Review workload and work-life balance

    • Increase employee engagement initiatives
    """)

    else:

        st.error("""
    ### 🚨 Recommendations

    • Immediate retention discussion required

    • Review compensation and benefits

    • Provide career growth opportunities

    • Address workplace concerns proactively

    • Create personalized retention plan
    """)

    st.markdown("---")

    st.markdown("""
<div style='text-align:center; color:gray;'>

Employee Attrition Intelligence System © 2026

Built with Streamlit & Machine Learning

</div>
""", unsafe_allow_html=True)
