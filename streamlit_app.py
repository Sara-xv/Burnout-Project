"""
Burnout Risk Analytics App — Streamlit

Three pages:
    1. Overview       — workforce stats + key EDA findings (cards + tables)
    2. What-If Simulator — HR enters/adjusts one employee's profile and
       gets a predicted burnout risk, using the trained Random Forest
       model with the same role-based custom thresholds used in
       burnout_prediction_model.py
    3. Model Performance — classification report + feature importance for
       the trained model, so the predictions above can be trusted/audited

Run with:
    streamlit run streamlit_app.py

Expects, relative to this file:
    Dataset/dataset_clean.csv          (the cleaned dataset)
    Model/burnout_risk_model.joblib    (the trained model from
                                         burnout_prediction_model.py)

Theme matches the rest of the project: dark background, navy/beige/cream
palette, green/yellow/red heat coloring for Low/Moderate/High.
"""

import joblib
import pandas as pd
import streamlit as st

# =========================================================================
# CONFIG & THEME
# =========================================================================
st.set_page_config(
    page_title="Burnout Risk Analytics",
    page_icon="🧠",
    layout="wide",
)

DATA_PATH = "Dataset/dataset_clean.csv"
MODEL_PATH = "Model/burnout_risk_model.joblib"

COLOR_NAVY = "#333F6E"
COLOR_BLUE_GREY = "#869EAF"
COLOR_BEIGE = "#DFD097"
COLOR_CREAM = "#CECEAB"
COLOR_GREEN = "#A1E197"
COLOR_RED = "#E08D8D"

CUSTOM_CSS = f"""
<style>
.stApp {{
    background-color: #000000;
    color: {COLOR_CREAM};
}}
h1, h2, h3 {{
    color: #FFFFFF !important;
}}
[data-testid="stMetricValue"] {{
    color: {COLOR_BEIGE} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {COLOR_BLUE_GREY} !important;
}}
.risk-badge {{
    display: inline-block;
    padding: 10px 24px;
    border-radius: 8px;
    font-size: 1.3em;
    font-weight: 700;
}}
.risk-high {{ background-color: rgba(224, 141, 141, 0.5); color: #fff; }}
.risk-moderate {{ background-color: rgba(232, 213, 139, 0.45); color: #111; }}
.risk-low {{ background-color: rgba(161, 225, 151, 0.35); color: #111; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================================================================
# DATA & MODEL LOADING (cached so the app stays fast)
# =========================================================================
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


data = load_data()
model = load_model()

# Feature columns the model was trained on (must match training order)
MODEL_FEATURE_COLUMNS = [
    'age', 'years_in_company', 'hours_worked_per_week', 'work_life_balance_score',
    'job_satisfaction_score', 'stress_level_self_report', 'anxiety_score',
    'depression_score', 'sleep_quality_score', 'social_support_score',
    'presenteeism_days_per_month', 'left_company_last_year', 'commute_time_mins',
    'monthly_salary_usd', 'manager_support_score', 'workload_index',
    'vacation_days_taken', 'years_since_last_promotion', 'absenteeism_days_per_month',
    'overtime_hours_per_week', 'mental_health_benefit_utilized',
    'gender_Male', 'gender_Non-binary',
    'role_Executive', 'role_HR', 'role_Manager', 'role_Sales', 'role_Support',
    'work_mode_On-site', 'work_mode_Remote',
]

SENSITIVE_ROLES = {'Manager', 'Executive'}
HIGH_THRESHOLD_SENSITIVE = 0.20
HIGH_THRESHOLD_DEFAULT = 0.25
LOW_THRESHOLD = 0.30

CLASS_ORDER = list(model.classes_)
HIGH_IDX = CLASS_ORDER.index('High')
LOW_IDX = CLASS_ORDER.index('Low')


def predict_with_custom_threshold(probs_row, role):
    """probs_row: array of P(High), P(Low), P(Moderate) in model.classes_ order."""
    high_th = HIGH_THRESHOLD_SENSITIVE if role in SENSITIVE_ROLES else HIGH_THRESHOLD_DEFAULT
    if probs_row[HIGH_IDX] >= high_th:
        return 'High'
    if probs_row[LOW_IDX] >= LOW_THRESHOLD:
        return 'Low'
    return 'Moderate'


def risk_badge_html(risk_label: str) -> str:
    cls = {"High": "risk-high", "Moderate": "risk-moderate", "Low": "risk-low"}[risk_label]
    return f'<span class="risk-badge {cls}">{risk_label} Risk</span>'


# =========================================================================
# RECOMMENDATION ENGINE
# =========================================================================
# Benchmarks = median value of each factor among employees who are
# actually Low-risk in the training data. These are the evidence-based
# "healthy" reference points every recommendation is measured against.
# (Computed once from work_mental_health.csv; hardcoded here so the app
# doesn't need the full dataset in memory just to generate advice.)
RISK_BENCHMARKS = {
    'workload_index': {
        'target': 2.9, 'direction': 'lower', 'unit': 'pts (0-10 scale)',
        'label': 'Workload Index',
        'why': "Workload Index is the 2nd strongest driver of burnout risk in this dataset "
               "(Pearson r=+0.44). High-risk employees have a median of 5.0 vs 2.9 for Low-risk.",
    },
    'overtime_hours_per_week': {
        'target': 0.0, 'direction': 'lower', 'unit': 'hrs/week',
        'label': 'Overtime Hours',
        'why': "Overtime shows a moderate positive correlation with burnout risk (r=+0.29). "
               "Low-risk employees typically log 0 overtime hours; High-risk employees log 2+.",
    },
    'hours_worked_per_week': {
        'target': 42.0, 'direction': 'lower', 'unit': 'hrs/week',
        'label': 'Hours Worked per Week',
        'why': "The single clearest pattern in the EDA: High-risk burnout jumps from 2.3% at "
               "30-39 hrs/week to 57.7% at 60+ hrs/week.",
    },
    'manager_support_score': {
        'target': 3.5, 'direction': 'higher', 'unit': 'pts (1-5 scale)',
        'label': 'Manager Support',
        'why': "Interaction analysis found manager support acts as a real buffer: under high "
               "workload, low support employees reach 87.5% High-risk vs 70.7% with high support.",
    },
    'sleep_quality_score': {
        'target': 4.1, 'direction': 'higher', 'unit': 'pts (1-5 scale)',
        'label': 'Sleep Quality',
        'why': "Sleep Quality is one of the two strongest protective factors found (r=-0.24).",
    },
    'work_life_balance_score': {
        'target': 4.2, 'direction': 'higher', 'unit': 'pts (1-5 scale)',
        'label': 'Work-Life Balance',
        'why': "The single strongest protective factor found in the correlation analysis (r=-0.31).",
    },
}

# Priority order: which factor to recommend fixing first when several are
# off-target at once. Ordered by effect size found in the EDA (strongest
# risk drivers and protective factors first).
RECOMMENDATION_PRIORITY = [
    'hours_worked_per_week', 'workload_index', 'overtime_hours_per_week',
    'work_life_balance_score', 'sleep_quality_score', 'manager_support_score',
]

# Concrete, step-by-step actions for each factor (shown for whichever
# factors are flagged as off-target for this specific employee).
RECOMMENDATION_ACTIONS = {
    'hours_worked_per_week': [
        "Cap scheduled hours at 45-50/week for the next 2-4 weeks and track actual logged hours.",
        "Identify 1-2 recurring tasks that can be delegated, automated, or dropped entirely.",
        "Review project deadlines with the employee's manager — unrealistic timelines are a common root cause.",
    ],
    'workload_index': [
        "Audit current open tasks/tickets assigned to this employee vs. the team average.",
        "Redistribute at least one active project to a teammate with lower workload.",
        "Re-check workload_index after 2-4 weeks to confirm the reduction actually took effect.",
    ],
    'overtime_hours_per_week': [
        "Set a hard cap (e.g. max 5 hrs/week) and require manager sign-off above that.",
        "Investigate whether overtime is driven by understaffing, scope creep, or poor estimation — fix the root cause, not just the symptom.",
    ],
    'work_life_balance_score': [
        "Confirm the employee is taking their full entitled vacation/PTO — check vacation_days_taken.",
        "Introduce or enforce 'no meetings' blocks or a communication cutoff outside working hours.",
        "Revisit with the employee in 30 days using a short pulse survey to check if balance improved.",
    ],
    'sleep_quality_score': [
        "This is a personal health indicator — HR's role is to remove work-driven causes (late meetings, "
        "on-call stress, overtime) rather than prescribe personal habits.",
        "If overtime/hours are already high, fixing those first often improves sleep quality indirectly.",
    ],
    'manager_support_score': [
        "Schedule a 1:1 check-in between the employee and their manager within the next 1-2 weeks.",
        "Review manager workload — overloaded managers often provide less support to their reports.",
        "Consider manager coaching/training if this pattern repeats across multiple reports.",
    ],
}


def generate_recommendations(profile: dict, role: str, predicted_risk: str) -> list:
    """
    Compare the employee's input values against evidence-based benchmarks
    (median of Low-risk employees) and return a prioritized, step-by-step
    list of recommendations for factors that are off-target.
    Returns a list of dicts: {label, current, target, gap, unit, why, actions}
    """
    if predicted_risk == 'Low':
        return []

    recommendations = []
    for feature in RECOMMENDATION_PRIORITY:
        bench = RISK_BENCHMARKS[feature]
        current_value = profile[feature]
        target = bench['target']

        is_off_target = (
            (bench['direction'] == 'lower' and current_value > target) or
            (bench['direction'] == 'higher' and current_value < target)
        )
        if not is_off_target:
            continue

        gap = abs(current_value - target)
        recommendations.append({
            'feature': feature,
            'label': bench['label'],
            'current': current_value,
            'target': target,
            'direction': bench['direction'],
            'gap': gap,
            'unit': bench['unit'],
            'why': bench['why'],
            'actions': RECOMMENDATION_ACTIONS[feature],
        })

    # Role-specific note: Manager/Executive roles showed the strongest
    # sensitivity to manager_support_score in the interaction analysis.
    return recommendations


# =========================================================================
# SIDEBAR NAVIGATION
# =========================================================================
st.sidebar.title("🧠 Burnout Risk Analytics")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🔮 What-If Simulator", "📈 Model Performance"],
)

# =========================================================================
# PAGE 1 — OVERVIEW
# =========================================================================
def render_overview():
    st.title("📊 Workforce Overview")
    st.caption(f"{len(data):,} employees — key stats and burnout risk patterns from the EDA.")

    # ---- Row 1: headline KPIs ----
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Employees", f"{len(data):,}")
    col2.metric("Avg Hours/Week", f"{data['hours_worked_per_week'].mean():.1f}")
    col3.metric("Avg Salary", f"${data['monthly_salary_usd'].mean():,.0f}")
    col4.metric("High-Risk Rate", f"{(data['burnout_risk']=='High').mean()*100:.1f}%")
    col5.metric("Turnover (last yr)", f"{data['left_company_last_year'].mean()*100:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Burnout Risk Distribution")
        risk_counts = data['burnout_risk'].value_counts().reindex(['Low', 'Moderate', 'High'])
        st.bar_chart(risk_counts)

        st.subheader("By Role")
        role_risk = pd.crosstab(data['role'], data['burnout_risk'], normalize='index') * 100
        role_risk = role_risk[['Low', 'Moderate', 'High']].round(1)
        st.dataframe(role_risk, use_container_width=True)

        st.subheader("By Work Mode")
        mode_risk = pd.crosstab(data['work_mode'], data['burnout_risk'], normalize='index') * 100
        mode_risk = mode_risk[['Low', 'Moderate', 'High']].round(1)
        st.dataframe(mode_risk, use_container_width=True)

    with right:
        st.subheader("Burnout Risk by Working Hours")
        hours_bins = [29, 39, 49, 59, 70]
        hours_labels = ['30-39', '40-49', '50-59', '60+']
        hours_group = pd.cut(data['hours_worked_per_week'], bins=hours_bins, labels=hours_labels)
        hours_high_pct = data.groupby(hours_group, observed=False)['burnout_risk'].apply(
            lambda x: (x == 'High').mean() * 100
        )
        st.bar_chart(hours_high_pct)
        st.caption("% of employees at High risk, by weekly hours worked.")

        st.subheader("Mental Health Benefit Usage")
        b1, b2 = st.columns(2)
        b1.metric("Benefit Utilization Rate", f"{data['mental_health_benefit_utilized'].mean()*100:.1f}%")
        used_high_pct = data[data['mental_health_benefit_utilized'] == 1]['burnout_risk'].eq('High').mean() * 100
        not_used_high_pct = data[data['mental_health_benefit_utilized'] == 0]['burnout_risk'].eq('High').mean() * 100
        b2.metric("High-Risk: Used vs Not", f"{used_high_pct:.0f}% / {not_used_high_pct:.0f}%")

    st.divider()
    st.subheader("⚠️ Where Risk Concentrates Most (Interaction Effects)")
    st.caption("Single-factor averages hide how much worse things get when risk factors stack together.")

    i1, i2 = st.columns(2)
    with i1:
        st.markdown("**Manager Support × Workload** (% High Risk)")
        support_bins = [0, 2, 3, 4, 6]
        support_labels = ['Low', 'Mid', 'Good', 'High']
        workload_bins = [-1, 3, 5, 7, 11]
        workload_labels = ['Low (0-3)', 'Mid-Low (3-5)', 'Mid-High (5-7)', 'High (7+)']
        support_group = pd.cut(data['manager_support_score'], bins=support_bins, labels=support_labels, right=False)
        workload_group = pd.cut(data['workload_index'], bins=workload_bins, labels=workload_labels)
        pivot1 = data.pivot_table(
            index=support_group, columns=workload_group, values='burnout_risk',
            aggfunc=lambda x: (x == 'High').mean() * 100, observed=False
        )
        st.dataframe(pivot1.round(1), use_container_width=True)

    with i2:
        st.markdown("**Role × Hours Worked/Week** (% High Risk)")
        hours_bins2 = [29, 39, 49, 59, 70]
        hours_labels2 = ['30-39', '40-49', '50-59', '60+']
        hours_group2 = pd.cut(data['hours_worked_per_week'], bins=hours_bins2, labels=hours_labels2)
        pivot2 = data.pivot_table(
            index='role', columns=hours_group2, values='burnout_risk',
            aggfunc=lambda x: (x == 'High').mean() * 100, observed=False
        )
        st.dataframe(pivot2.round(1), use_container_width=True)

    st.markdown(f"""
    - 🔴 **Stress Level**, **Workload Index**, **Hours Worked** — strongest risk drivers (correlation analysis)
    - 🟢 **Work-Life Balance**, **Sleep Quality** — strongest protective factors
    - 🔴 Low manager support + high workload → **87.5%** high-risk rate (vs 70.7% with high support)
    - 🔴 At 60+ hrs/week, Managers reach **73.8%** High-risk while Support staff stay at **29.4%**
    """)

print("Page 1 (Overview) ready.")

# =========================================================================
# PAGE 2 — WHAT-IF SIMULATOR
# =========================================================================
def render_simulator():
    st.title("🔮 What-If Simulator")
    st.caption("Enter an employee's profile (or adjust an existing one) to see their predicted burnout risk.")

    with st.form("employee_form"):
        st.subheader("Employee Profile")

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 22, 60, 35)
            gender = st.selectbox("Gender", ["Male", "Female", "Non-binary"])
            role = st.selectbox("Role", ["Developer", "Executive", "HR", "Manager", "Sales", "Support"])
            work_mode = st.selectbox("Work Mode", ["Hybrid", "On-site", "Remote"])
            years_in_company = st.slider("Years in Company", 0, 30, 5)
            monthly_salary = st.slider("Monthly Salary (USD)", 2500, 19450, 8000, step=100)

        with c2:
            hours_worked = st.slider("Hours Worked / Week", 30, 70, 45)
            overtime_hours = st.slider("Overtime Hours / Week", 0, 25, 5)
            commute_time = st.slider("Commute Time (mins)", 0, 120, 30)
            workload_index = st.slider("Workload Index", 0.0, 10.0, 4.0, step=0.1)
            manager_support = st.slider("Manager Support Score", 1.0, 5.0, 3.0, step=0.1)
            social_support = st.slider("Social Support Score", 1.0, 5.0, 3.0, step=0.1)

        with c3:
            work_life_balance = st.slider("Work-Life Balance Score", 1.0, 5.0, 3.5, step=0.1)
            sleep_quality = st.slider("Sleep Quality Score", 1.0, 5.0, 3.5, step=0.1)
            job_satisfaction = st.slider("Job Satisfaction Score", 1.0, 5.0, 3.5, step=0.1)
            stress_level = st.slider("Stress Level (self-report)", 1.0, 5.0, 2.5, step=0.1)
            anxiety_score = st.slider("Anxiety Score", 0.0, 10.0, 3.5, step=0.1)
            depression_score = st.slider("Depression Score", 0.0, 10.0, 3.0, step=0.1)

        with st.expander("Additional fields (defaults are reasonable averages)"):
            d1, d2 = st.columns(2)
            with d1:
                presenteeism = st.slider("Presenteeism (days/month)", 0, 10, 2)
                absenteeism = st.slider("Absenteeism (days/month)", 0, 10, 1)
                vacation_days = st.slider("Vacation Days Taken", 0, 30, 12)
            with d2:
                years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
                left_company = st.checkbox("Left Company Last Year", value=False)
                benefit_used = st.checkbox("Used Mental Health Benefit", value=False)

        submitted = st.form_submit_button("Predict Burnout Risk", type="primary")

    if submitted:
        # Build a single-row DataFrame matching the model's expected columns
        row = {col: 0 for col in MODEL_FEATURE_COLUMNS}
        row.update({
            'age': age,
            'years_in_company': years_in_company,
            'hours_worked_per_week': hours_worked,
            'work_life_balance_score': work_life_balance,
            'job_satisfaction_score': job_satisfaction,
            'stress_level_self_report': stress_level,
            'anxiety_score': anxiety_score,
            'depression_score': depression_score,
            'sleep_quality_score': sleep_quality,
            'social_support_score': social_support,
            'presenteeism_days_per_month': presenteeism,
            'left_company_last_year': int(left_company),
            'commute_time_mins': commute_time,
            'monthly_salary_usd': monthly_salary,
            'manager_support_score': manager_support,
            'workload_index': workload_index,
            'vacation_days_taken': vacation_days,
            'years_since_last_promotion': years_since_promotion,
            'absenteeism_days_per_month': absenteeism,
            'overtime_hours_per_week': overtime_hours,
            'mental_health_benefit_utilized': int(benefit_used),
        })
        if gender == "Male":
            row['gender_Male'] = 1
        elif gender == "Non-binary":
            row['gender_Non-binary'] = 1
        # Female is the drop_first baseline -> both gender dummies stay 0

        if role != "Developer":  # Developer is the drop_first baseline
            row[f'role_{role}'] = 1

        if work_mode != "Hybrid":  # Hybrid is the drop_first baseline
            row[f'work_mode_{work_mode}'] = 1

        X_input = pd.DataFrame([row])[MODEL_FEATURE_COLUMNS]

        probs = model.predict_proba(X_input)[0]
        default_pred = model.predict(X_input)[0]
        custom_pred = predict_with_custom_threshold(probs, role)

        st.divider()
        st.subheader("Prediction Result")

        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Default prediction** (highest-probability class):")
            st.markdown(risk_badge_html(default_pred), unsafe_allow_html=True)
        with r2:
            st.markdown("**Role-aware prediction** (custom threshold — recommended):")
            st.markdown(risk_badge_html(custom_pred), unsafe_allow_html=True)

        st.caption(
            f"P(Low) = {probs[LOW_IDX]:.2f}  |  "
            f"P(Moderate) = {probs[CLASS_ORDER.index('Moderate')]:.2f}  |  "
            f"P(High) = {probs[HIGH_IDX]:.2f}"
        )

        # ---- Recommendations ----
        st.divider()
        st.subheader("📋 Recommended Actions")

        recs = generate_recommendations(row, role, custom_pred)

        if custom_pred == "Low":
            st.success(
                "✅ No action needed. This profile's key indicators are already at or better than "
                "the benchmark for Low-risk employees in this dataset."
            )
        elif not recs:
            st.info(
                "This profile is flagged "
                f"**{custom_pred}** risk, but no single factor is clearly off-target against the "
                "Low-risk benchmark — risk here likely comes from a combination of smaller factors "
                "rather than one dominant cause. Consider a broader wellbeing check-in."
            )
        else:
            st.caption(
                f"Based on comparing this profile against the median values of employees who are "
                f"actually Low-risk in this dataset. Ordered by expected impact (strongest EDA "
                f"effect sizes first)."
            )

            if role in SENSITIVE_ROLES:
                st.markdown(
                    f"ℹ️ **{role}** roles showed the highest sensitivity to manager support in the "
                    "interaction analysis (Manager: 37.0% High-risk with low support vs 16.5% with "
                    "high support) — prioritize the Manager Support recommendation below if present."
                )

            for i, rec in enumerate(recs, start=1):
                direction_word = "Reduce" if rec['direction'] == 'lower' else "Increase"
                arrow = "🔽" if rec['direction'] == 'lower' else "🔼"

                with st.expander(
                    f"{i}. {arrow} {rec['label']} — {direction_word} by {rec['gap']:.1f} {rec['unit']}",
                    expanded=(i == 1),
                ):
                    gcol1, gcol2, gcol3 = st.columns(3)
                    gcol1.metric("Current", f"{rec['current']:.1f}")
                    gcol2.metric("Target (Low-risk benchmark)", f"{rec['target']:.1f}")
                    gcol3.metric("Gap", f"{rec['gap']:.1f} {rec['unit']}")

                    st.markdown(f"**Why this matters:** {rec['why']}")
                    st.markdown("**Step-by-step actions:**")
                    for action in rec['actions']:
                        st.markdown(f"- {action}")

            st.caption(
                "⚕️ These are HR/management-level process recommendations based on dataset patterns, "
                "not medical advice. For employees showing signs of acute distress, involve HR and, "
                "where appropriate, a mental health professional directly."
            )

print("Page 2 (What-If Simulator) ready.")

# =========================================================================
# PAGE 3 — MODEL PERFORMANCE
# =========================================================================
@st.cache_data
def compute_test_performance():
    """
    Recomputes real classification performance on a held-out test split,
    for both default predictions and role-based custom thresholds — so
    these numbers always reflect the actual loaded model, not a hardcoded
    snapshot from a past training run.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    y = data['burnout_risk']
    X_raw = data.drop(columns=['employee_id', 'burnout_risk'])
    X = pd.get_dummies(X_raw, columns=['gender', 'role', 'work_mode'], drop_first=True)
    X = X[MODEL_FEATURE_COLUMNS]  # enforce exact column order the model expects

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    role_test = data.loc[X_test.index, 'role']

    y_pred_default = model.predict(X_test)
    report_default = classification_report(y_test, y_pred_default, output_dict=True)

    probs = model.predict_proba(X_test)
    y_pred_custom = [
        predict_with_custom_threshold(p, role) for p, role in zip(probs, role_test.values)
    ]
    report_custom = classification_report(y_test, y_pred_custom, output_dict=True)

    return report_default, report_custom


def report_to_table(report: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "Class": ["High", "Low", "Moderate"],
        "Precision": [report[c]['precision'] for c in ["High", "Low", "Moderate"]],
        "Recall": [report[c]['recall'] for c in ["High", "Low", "Moderate"]],
        "F1-score": [report[c]['f1-score'] for c in ["High", "Low", "Moderate"]],
    }).set_index("Class").round(2)


def render_model_performance():
    st.title("📈 Model Performance")
    st.caption("Random Forest classifier trained to predict burnout_risk. Metrics below are computed live on a held-out 20% test set.")

    st.subheader("Why Custom Thresholds")
    st.markdown("""
    burnout_risk is imbalanced (Moderate=66%, Low=18%, High=16%). With default predictions
    (highest-probability class wins), the model mostly predicts "Moderate" and misses most real
    High-risk employees. Custom thresholds — flagging High if P(High) exceeds a threshold, even
    when it isn't the single most likely class — trade a little overall accuracy for much better
    recall on the classes that matter most to HR (missing a real High-risk employee is costlier
    than an extra false alarm).
    """)

    st.subheader("Role-Based Thresholds")
    st.markdown(f"""
    - **Manager / Executive** (highest actual High-risk rates: 25.7% / 20.3%): flagged High if P(High) ≥ {HIGH_THRESHOLD_SENSITIVE}
    - **All other roles**: flagged High if P(High) ≥ {HIGH_THRESHOLD_DEFAULT}
    - Flagged Low if P(Low) ≥ {LOW_THRESHOLD}, otherwise Moderate
    """)

    st.divider()
    st.subheader("Performance (live, on held-out test set)")

    report_default, report_custom = compute_test_performance()

    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.markdown("**Default predictions**")
        st.table(report_to_table(report_default))
        st.metric("Overall Accuracy", f"{report_default['accuracy']*100:.1f}%")
    with perf_col2:
        st.markdown("**Role-based custom thresholds**")
        st.table(report_to_table(report_custom))
        st.metric("Overall Accuracy", f"{report_custom['accuracy']*100:.1f}%")

    high_before = report_default['High']['recall'] * 100
    high_after = report_custom['High']['recall'] * 100
    st.info(
        f"High-risk recall improved from **{high_before:.0f}%** to **{high_after:.0f}%** using custom "
        f"thresholds — the model now catches roughly {high_after/max(high_before,1):.1f}x as many real "
        f"High-risk employees, at the cost of some extra false alarms and a "
        f"{(report_default['accuracy']-report_custom['accuracy'])*100:.1f} point drop in overall accuracy."
    )

    st.divider()
    st.subheader("Feature Importance")
    st.caption("How much each feature contributed to the model's predictions overall (Gini importance).")
    importances = pd.Series(model.feature_importances_, index=MODEL_FEATURE_COLUMNS).sort_values(ascending=False)
    st.bar_chart(importances.head(15))

    with st.expander("How to read this page"):
        st.markdown("""
        - **Precision**: of everyone the model flagged as a given class, what % actually were that class.
        - **Recall**: of everyone who actually was a given class, what % the model correctly caught.
        - For HR, **recall on "High"** matters most — it measures how many real at-risk employees
          would NOT slip through undetected.
        - These numbers are evaluated on a 20% test set the model never saw during training, so they
          reflect realistic performance rather than memorization.
        """)

print("Page 3 (Model Performance) ready.")

# =========================================================================
# PAGE ROUTER
# =========================================================================
if page == "📊 Overview":
    render_overview()
elif page == "🔮 What-If Simulator":
    render_simulator()
elif page == "📈 Model Performance":
    render_model_performance()