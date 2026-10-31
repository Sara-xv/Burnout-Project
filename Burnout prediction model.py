import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

INPUT_PATH = "Dataset/dataset_clean.csv"
MODEL_OUTPUT_PATH = "Model/burnout_risk_model.joblib"
HTML_OUTPUT_PATH = "Report/burnout_prediction_model.html"

os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(HTML_OUTPUT_PATH), exist_ok=True)

# ---- Theme (same palette as the rest of the project) ----
COLOR_BG = "#000000"
COLOR_PANEL_BG = "#0D0D12"
COLOR_TABLE_BG = "#111111"
COLOR_NAVY = "#333F6E"
COLOR_BLUE_GREY = "#869EAF"
COLOR_BEIGE = "#DFD097"
COLOR_CREAM = "#CECEAB"
COLOR_GREEN = "#A1E197"
COLOR_RED = "#E08D8D"
COLOR_TEXT = "#FFFFFF"

CSS = f"""
body {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; font-family: 'Segoe UI', Arial, sans-serif; margin:0; padding:0; }}
.report-header {{ text-align:center; padding:50px 20px 30px 20px; border-bottom:2px solid {COLOR_NAVY}; }}
.report-header h1 {{ color:{COLOR_GREEN}; font-size:2.0em; }}
.section {{ max-width:1100px; margin:0 auto 60px auto; padding:0 20px; }}
.section h2 {{ color:{COLOR_TEXT}; border-left:5px solid {COLOR_GREEN}; padding-left:14px; font-size:1.4em; }}
.section .section-desc {{ color:{COLOR_BLUE_GREY}; margin-bottom:20px; margin-left:19px; }}
.subsection h3 {{ color:{COLOR_BEIGE}; margin-top:30px; margin-bottom:10px; font-size:1.1em; }}
.table-container {{ overflow-x:auto; margin-bottom:15px; }}
table {{ width:100%; border-collapse:collapse; background-color:{COLOR_TABLE_BG}; border-radius:6px; overflow:hidden; }}
th {{ background-color:{COLOR_NAVY}; color:{COLOR_TEXT}; padding:12px 14px; text-align:center; border-bottom:2px solid {COLOR_BLUE_GREY}; font-size:0.92em; }}
td {{ padding:10px 14px; text-align:center; border-bottom:1px solid #262626; color:{COLOR_CREAM}; font-size:0.92em; }}
tr:hover td {{ filter:brightness(1.2); }}
.stat-row {{ display:flex; flex-wrap:wrap; gap:16px; margin-bottom:15px; }}
.stat-card {{ background-color:{COLOR_PANEL_BG}; border:1px solid {COLOR_NAVY}; border-radius:8px; padding:16px 22px; flex:1; min-width:160px; }}
.stat-card .label {{ color:{COLOR_BLUE_GREY}; font-size:0.85em; margin-bottom:6px; }}
.stat-card .value {{ color:{COLOR_BEIGE}; font-size:1.4em; font-weight:700; }}
.takeaway-panel {{ background-color:{COLOR_PANEL_BG}; border:1px solid {COLOR_GREEN}; border-radius:8px; padding:20px 26px; margin-top:20px; }}
.takeaway-panel .title {{ color:{COLOR_GREEN}; font-weight:700; margin-bottom:10px; }}
.takeaway-panel ul {{ color:{COLOR_CREAM}; margin:0; padding-left:20px; line-height:1.8; }}
.takeaway-panel b {{ color:{COLOR_BEIGE}; }}
footer {{ text-align:center; color:{COLOR_BLUE_GREY}; padding:30px; font-size:0.85em; border-top:1px solid {COLOR_NAVY}; }}
"""


def stat_cards(items: dict) -> str:
    cards = "".join(
        f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div></div>'
        for label, value in items.items()
    )
    return f'<div class="stat-row">{cards}</div>'


# =========================================================================
# 1. Load and prepare data
# =========================================================================
data = pd.read_csv(INPUT_PATH)

y = data['burnout_risk']
X_raw = data.drop(columns=['employee_id', 'burnout_risk'])
X = pd.get_dummies(X_raw, columns=['gender', 'role', 'work_mode'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
role_test = data.loc[X_test.index, 'role']

# =========================================================================
# 2. Train Random Forest
# =========================================================================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

joblib.dump(model, MODEL_OUTPUT_PATH)
print(f"Model saved to {MODEL_OUTPUT_PATH}")

# =========================================================================
# 3. Baseline predictions (default: highest-probability class wins)
# =========================================================================
y_pred_default = model.predict(X_test)
report_default = classification_report(y_test, y_pred_default, output_dict=True)

# =========================================================================
# 4. Role-based custom threshold predictions
# =========================================================================
classes = model.classes_
probs = model.predict_proba(X_test)
high_idx = list(classes).index('High')
low_idx = list(classes).index('Low')

SENSITIVE_ROLES = {'Manager', 'Executive'}
HIGH_THRESHOLD_SENSITIVE = 0.20
HIGH_THRESHOLD_DEFAULT = 0.25
LOW_THRESHOLD = 0.30


def role_based_predict(probs, roles):
    preds = []
    for p, role in zip(probs, roles):
        high_th = HIGH_THRESHOLD_SENSITIVE if role in SENSITIVE_ROLES else HIGH_THRESHOLD_DEFAULT
        if p[high_idx] >= high_th:
            preds.append('High')
        elif p[low_idx] >= LOW_THRESHOLD:
            preds.append('Low')
        else:
            preds.append('Moderate')
    return preds


y_pred_custom = role_based_predict(probs, role_test.values)
report_custom = classification_report(y_test, y_pred_custom, output_dict=True)

print("\n=== Default predictions (highest-probability class) ===")
print(classification_report(y_test, y_pred_default))
print("\n=== Role-based custom threshold predictions ===")
print(classification_report(y_test, y_pred_custom))

# =========================================================================
# 5. Feature importance
# =========================================================================
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

FEATURE_LABELS = {
    "stress_level_self_report": "Stress Level (self-report)",
    "workload_index": "Workload Index",
    "hours_worked_per_week": "Hours Worked per Week",
    "overtime_hours_per_week": "Overtime (hrs/week)",
    "anxiety_score": "Anxiety Score",
    "depression_score": "Depression Score",
    "monthly_salary_usd": "Monthly Salary",
    "commute_time_mins": "Commute Time",
    "presenteeism_days_per_month": "Presenteeism (days/month)",
    "mental_health_benefit_utilized": "Mental Health Benefit Used",
    "left_company_last_year": "Left Company Last Year",
    "absenteeism_days_per_month": "Absenteeism (days/month)",
    "years_in_company": "Years in Company",
    "years_since_last_promotion": "Years Since Last Promotion",
    "age": "Age",
    "social_support_score": "Social Support",
    "vacation_days_taken": "Vacation Days Taken",
    "manager_support_score": "Manager Support",
    "job_satisfaction_score": "Job Satisfaction",
    "sleep_quality_score": "Sleep Quality",
    "work_life_balance_score": "Work-Life Balance",
}


def pretty_feature_name(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    # one-hot encoded columns like role_Manager, gender_Male, work_mode_Remote
    for prefix, pretty_prefix in [("role_", "Role: "), ("gender_", "Gender: "), ("work_mode_", "Work Mode: ")]:
        if name.startswith(prefix):
            return pretty_prefix + name[len(prefix):]
    return name


# =========================================================================
# 6. Build HTML report
# =========================================================================
model_stats = stat_cards({
    "Model": "Random Forest (200 trees)",
    "Train / Test Split": "80% / 20%",
    "Overall Accuracy (default)": f"{report_default['accuracy']*100:.1f}%",
    "Overall Accuracy (custom thresholds)": f"{report_custom['accuracy']*100:.1f}%",
})


def classification_report_table(report: dict) -> str:
    rows = ""
    for label in ['Low', 'Moderate', 'High']:
        r = report[label]
        rows += (
            f"<tr><th>{label}</th><td>{r['precision']:.2f}</td>"
            f"<td>{r['recall']:.2f}</td><td>{r['f1-score']:.2f}</td>"
            f"<td>{int(r['support'])}</td></tr>"
        )
    return f"""
    <div class="table-container">
    <table>
        <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """


top_features_rows = "".join(
    f"<tr><th>{pretty_feature_name(name)}</th><td>{value:.3f}</td></tr>"
    for name, value in importances.head(15).items()
)
feature_table_html = f"""
<div class="table-container">
<table>
    <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
    <tbody>{top_features_rows}</tbody>
</table>
</div>
"""

high_recall_before = report_default['High']['recall'] * 100
high_recall_after = report_custom['High']['recall'] * 100

section_html = f"""
<div class="section" id="prediction-model">
    <h2>Burnout Risk Prediction Model</h2>
    <p class="section-desc">A Random Forest classifier trained to predict burnout_risk (Low / Moderate / High) from the rest of the dataset.</p>
    {model_stats}

    <div class="subsection">
        <h3>Why Custom Thresholds Were Needed</h3>
        <p style="color:{COLOR_CREAM}; margin-left:4px;">
            burnout_risk is imbalanced (Moderate=66%, Low=18%, High=16%). With default predictions,
            the model correctly identifies only {report_default['High']['recall']*100:.0f}% of real
            High-risk employees — it defaults to predicting "Moderate" too often. Missing a real
            High-risk employee is costlier than a false alarm, so custom decision thresholds were
            applied: an employee is flagged High if P(High) exceeds a threshold, even when it isn't
            the single most likely class.
        </p>
    </div>

    <div class="subsection">
        <h3>Role-Based Thresholds</h3>
        <p style="color:{COLOR_CREAM}; margin-left:4px;">
            Manager and Executive roles have the highest actual High-risk rates in this dataset
            (25.7% and 20.3% respectively, vs 11-17% for other roles — see earlier EDA), so they use
            a more sensitive threshold (P(High) ≥ {HIGH_THRESHOLD_SENSITIVE}) than other roles
            (P(High) ≥ {HIGH_THRESHOLD_DEFAULT}).
        </p>
    </div>

    <div class="subsection">
        <h3>Model Performance — Default Predictions</h3>
        {classification_report_table(report_default)}
    </div>

    <div class="subsection">
        <h3>Model Performance — Role-Based Custom Thresholds</h3>
        {classification_report_table(report_custom)}
    </div>

    <div class="subsection">
        <h3>Top 15 Most Important Features</h3>
        {feature_table_html}
    </div>

    <div class="takeaway-panel">
        <div class="title">Key Takeaways</div>
        <ul>
            <li><b>High-risk recall improved from {high_recall_before:.0f}% to {high_recall_after:.0f}%</b>
            by using custom thresholds instead of default predictions — at a small cost to overall accuracy
            ({report_default['accuracy']*100:.1f}% → {report_custom['accuracy']*100:.1f}%).</li>
            <li><b>Top feature:</b> {pretty_feature_name(importances.index[0])}
            (importance {importances.iloc[0]:.3f})</li>
            <li>Feature importance from this model can be compared against the earlier correlation
            analysis to confirm which factors matter most for burnout risk.</li>
        </ul>
    </div>
</div>
"""

header_html = """
<div class="report-header">
    <h1>🤖 Burnout Risk Prediction Model</h1>
</div>
"""

footer_html = """
<footer></footer>
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Burnout Risk Prediction Model</title>
<style>{CSS}</style>
</head>
<body>
{header_html}
{section_html}
{footer_html}
</body>
</html>
"""

with open(HTML_OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"\nSaved HTML report to {HTML_OUTPUT_PATH}")