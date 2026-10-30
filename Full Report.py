"""
Master HR Analytics Report — Burnout Risk Deep Dive

Consolidates every analysis produced so far into a single, consistently
themed HTML report:
    1. Data Quality Summary
    2. Workforce Overview (demographics, roles, work mode)
    3. Wellbeing & Workload Indicators
    4. Correlation with Burnout Risk (linear)
    5. Nonlinear Relationship with Burnout Risk (mutual information)
    6. Burnout Risk by Employee Profile (role, gender, work mode)
    7. Burnout Risk by Working Hours
    8. Burnout Risk by Tenure & Salary
    9. Burnout Risk by Company Support (turnover, mental health benefit)
    10. Interaction Effects (role x hours, mode x overtime, support x workload, role x support)
    11. Key Takeaways / Summary

Theme: upgraded version of the styled-table HTML theme (dark background,
navy headers, soft green/yellow/red heat coloring for Low/Moderate/High).
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

INPUT_PATH = "Dataset/dataset_clean.csv"
OUTPUT_PATH = "Report/master_hr_report.html"

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

data = pd.read_csv(INPUT_PATH)

# =========================================================================
# THEME — colors reused across every section
# =========================================================================
COLOR_BG = "#000000"
COLOR_PANEL_BG = "#0D0D12"
COLOR_TABLE_BG = "#111111"
COLOR_NAVY = "#333F6E"
COLOR_BLUE_GREY = "#869EAF"
COLOR_BEIGE = "#DFD097"
COLOR_CREAM = "#CECEAB"
COLOR_GREEN = "#A1E197"   # Low risk / good
COLOR_YELLOW = "#E8D58B"  # Moderate / neutral
COLOR_RED = "#E08D8D"     # High risk / bad
COLOR_TEXT = "#FFFFFF"

CSS = f"""
body {{
    background-color: {COLOR_BG};
    color: {COLOR_TEXT};
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: 0;
    padding: 0;
}}

.report-header {{
    text-align: center;
    padding: 50px 20px 30px 20px;
    border-bottom: 2px solid {COLOR_NAVY};
    margin-bottom: 10px;
}}

.report-header h1 {{
    color: {COLOR_GREEN};
    font-size: 2.2em;
    margin-bottom: 8px;
}}

.report-header p {{
    color: {COLOR_BLUE_GREY};
    font-size: 1.05em;
    max-width: 700px;
    margin: 0 auto;
}}

.toc {{
    max-width: 800px;
    margin: 30px auto 50px auto;
    background-color: {COLOR_PANEL_BG};
    border: 1px solid {COLOR_NAVY};
    border-radius: 8px;
    padding: 25px 35px;
}}

.toc h2 {{
    color: {COLOR_BEIGE};
    margin-top: 0;
}}

.toc ol {{
    color: {COLOR_CREAM};
    line-height: 1.9;
}}

.toc a {{
    color: {COLOR_CREAM};
    text-decoration: none;
}}

.toc a:hover {{
    color: {COLOR_GREEN};
    text-decoration: underline;
}}

.section {{
    max-width: 1100px;
    margin: 0 auto 60px auto;
    padding: 0 20px;
    scroll-margin-top: 20px;
}}

.section h2 {{
    color: {COLOR_TEXT};
    border-left: 5px solid {COLOR_GREEN};
    padding-left: 14px;
    font-size: 1.5em;
    margin-bottom: 6px;
}}

.section .section-desc {{
    color: {COLOR_BLUE_GREY};
    margin-bottom: 25px;
    margin-left: 19px;
}}

.subsection h3 {{
    color: {COLOR_BEIGE};
    margin-top: 35px;
    margin-bottom: 12px;
    font-size: 1.15em;
}}

.table-container {{
    overflow-x: auto;
    margin-bottom: 15px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    background-color: {COLOR_TABLE_BG};
    border-radius: 6px;
    overflow: hidden;
}}

th {{
    background-color: {COLOR_NAVY};
    color: {COLOR_TEXT};
    padding: 12px 14px;
    text-align: center;
    border-bottom: 2px solid {COLOR_BLUE_GREY};
    font-size: 0.92em;
}}

td {{
    padding: 10px 14px;
    text-align: center;
    border-bottom: 1px solid #262626;
    color: {COLOR_CREAM};
    font-size: 0.92em;
}}

tr:hover td {{
    filter: brightness(1.2);
}}

.cell-low {{ background-color: rgba(161, 225, 151, 0.28); color: #111 !important; font-weight: 600; }}
.cell-mod {{ background-color: rgba(232, 213, 139, 0.45); color: #111 !important; font-weight: 600; }}
.cell-high {{ background-color: rgba(224, 141, 141, 0.5); color: #fff !important; font-weight: 700; }}

.stat-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 15px;
}}

.stat-card {{
    background-color: {COLOR_PANEL_BG};
    border: 1px solid {COLOR_NAVY};
    border-radius: 8px;
    padding: 16px 22px;
    flex: 1;
    min-width: 160px;
}}

.stat-card .label {{
    color: {COLOR_BLUE_GREY};
    font-size: 0.85em;
    margin-bottom: 6px;
}}

.stat-card .value {{
    color: {COLOR_BEIGE};
    font-size: 1.4em;
    font-weight: 700;
}}

.takeaway-panel {{
    background-color: {COLOR_PANEL_BG};
    border: 1px solid {COLOR_GREEN};
    border-radius: 8px;
    padding: 20px 26px;
    margin-top: 20px;
}}

.takeaway-panel .title {{
    color: {COLOR_GREEN};
    font-weight: 700;
    margin-bottom: 10px;
    font-size: 1.05em;
}}

.takeaway-panel ul {{
    color: {COLOR_CREAM};
    margin: 0;
    padding-left: 20px;
    line-height: 1.8;
}}

.takeaway-panel b {{
    color: {COLOR_BEIGE};
}}

.final-summary {{
    max-width: 1100px;
    margin: 60px auto;
    padding: 30px 20px;
    border-top: 2px solid {COLOR_NAVY};
}}

.final-summary h2 {{
    color: {COLOR_GREEN};
    text-align: center;
    border: none;
    padding-left: 0;
}}

.final-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
    margin-top: 25px;
}}

.final-card {{
    background-color: {COLOR_PANEL_BG};
    border-radius: 8px;
    padding: 20px;
    border-top: 3px solid {COLOR_GREEN};
}}

.final-card.risk {{ border-top-color: {COLOR_RED}; }}
.final-card.protective {{ border-top-color: {COLOR_GREEN}; }}

.final-card h4 {{
    color: {COLOR_TEXT};
    margin-top: 0;
    margin-bottom: 10px;
}}

.final-card p {{
    color: {COLOR_CREAM};
    font-size: 0.92em;
    line-height: 1.6;
    margin: 0;
}}

footer {{
    text-align: center;
    color: {COLOR_BLUE_GREY};
    padding: 30px;
    font-size: 0.85em;
    border-top: 1px solid {COLOR_NAVY};
}}
"""

# =========================================================================
# HELPERS
# =========================================================================

def high_risk_pct(series: pd.Series) -> float:
    return (series == 'High').mean() * 100


def risk_cell_class(col_name: str) -> str:
    """Map a Low/Moderate/High column name to its heat-color CSS class."""
    name = col_name.strip().lower()
    if name == 'low':
        return 'cell-low'
    if name == 'moderate':
        return 'cell-mod'
    if name == 'high':
        return 'cell-high'
    return ''


def pct_heat_class(value: float) -> str:
    """For numeric % High-risk cells: green <15%, yellow 15-30%, red >30%."""
    if value >= 30:
        return 'cell-high'
    if value >= 15:
        return 'cell-mod'
    return 'cell-low'


def df_to_distribution_table(df: pd.DataFrame, index_col: str, count_label: str) -> str:
    """Build an HTML table: Group | Employees | Low% | Moderate% | High%"""
    grouped = df.groupby(index_col, observed=False)
    counts = grouped.size()
    pct = grouped['burnout_risk'].value_counts(normalize=True).unstack().fillna(0) * 100
    pct = pct[['Low', 'Moderate', 'High']] if set(['Low', 'Moderate', 'High']).issubset(pct.columns) else pct

    rows_html = ""
    for idx in counts.index:
        row_cells = f"<td>{counts[idx]:,}</td>"
        for col in pct.columns:
            val = pct.loc[idx, col]
            cls = risk_cell_class(col)
            row_cells += f'<td class="{cls}">{val:.1f}%</td>'
        rows_html += f"<tr><th>{idx}</th>{row_cells}</tr>\n"

    header_cells = "".join(f"<th>{c}</th>" for c in pct.columns)
    html = f"""
    <div class="table-container">
    <table>
        <thead>
            <tr><th>{count_label}</th><th>Employees</th>{header_cells}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    """
    return html


def pivot_to_heat_table(pivot: pd.DataFrame, row_label: str) -> str:
    """Build an HTML table from a role/group x group % High-risk pivot."""
    header_cells = "".join(f"<th>{c}</th>" for c in pivot.columns)
    rows_html = ""
    for idx, row in pivot.iterrows():
        cells = ""
        for val in row:
            if pd.isna(val):
                cells += "<td>—</td>"
            else:
                cls = pct_heat_class(val)
                cells += f'<td class="{cls}">{val:.1f}%</td>'
        rows_html += f"<tr><th>{idx}</th>{cells}</tr>\n"

    html = f"""
    <div class="table-container">
    <table>
        <thead>
            <tr><th>{row_label}</th>{header_cells}</tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>
    """
    return html


def stat_cards(items: dict) -> str:
    """items: {label: value_string}"""
    cards = "".join(
        f'<div class="stat-card"><div class="label">{label}</div>'
        f'<div class="value">{value}</div></div>'
        for label, value in items.items()
    )
    return f'<div class="stat-row">{cards}</div>'


print("Helpers loaded. Computing sections...")

# =========================================================================
# SECTION 1 — Data Quality Summary
# =========================================================================
missing_cols = [
    'work_life_balance_score', 'sleep_quality_score', 'manager_support_score',
    'years_since_last_promotion', 'monthly_salary_usd', 'anxiety_score',
    'job_satisfaction_score', 'social_support_score', 'stress_level_self_report',
    'depression_score'
]
# (Missing values were already imputed upstream with median imputation;
#  this section documents what was found and fixed, plus known false positives.)

section1_stats = stat_cards({
    "Total Rows": f"{len(data):,}",
    "Total Columns": f"{data.shape[1]}",
    "Columns w/ Missing (pre-cleaning)": f"{len(missing_cols)}",
    "Missing Rate Range": "0.77% – 1.68%",
})

section1_html = f"""
<div class="section" id="data-quality">
    <h2>1. Data Quality Summary</h2>
    <p class="section-desc">What was found in the raw dataset, what was fixed, and what was a false alarm.</p>
    {section1_stats}
    <div class="subsection">
        <h3>Missing Values — Real Issue, Resolved</h3>
        <p style="color:{COLOR_CREAM}; margin-left:4px;">
            10 numeric columns had missing values ranging from 0.77% to 1.68% per column
            (2,922 of 25,000 rows affected, 0.48% of all cells). The pattern was checked and found
            to be random (MCAR) — not concentrated in any particular role, work mode, or burnout
            risk group. Resolved via median imputation.
        </p>
    </div>
    <div class="subsection">
        <h3>Invalid Values — False Positive (No Action Needed)</h3>
        <p style="color:{COLOR_CREAM}; margin-left:4px;">
            An automated profiling tool flagged 16,382 values in <code>presenteeism_days_per_month</code>
            and <code>absenteeism_days_per_month</code> as "invalid" using a rule that expected values
            between 1 and 12 (as if these were calendar months). These columns actually represent a
            <i>count of days per month</i>, not a month number — the rule was misapplied due to the
            <code>_per_month</code> suffix in the column name. No correction needed.
        </p>
    </div>
    <div class="subsection">
        <h3>Outliers — Mostly Expected</h3>
        <p style="color:{COLOR_CREAM}; margin-left:4px;">
            16 of 21 numeric columns had statistical outliers flagged by IQR methods, but their
            contribution to the overall quality penalty was minor. Columns like
            <code>overtime_hours_per_week</code> (13.3% flagged) are naturally right-skewed —
            some employees genuinely work far more overtime than others. Not treated as defects.
        </p>
    </div>
</div>
"""

# =========================================================================
# SECTION 2 — Workforce Overview
# =========================================================================
age_stats = data['age'].agg(['min', 'mean', 'max'])
gender_counts = data['gender'].value_counts()
gender_pcts = data['gender'].value_counts(normalize=True) * 100
role_counts = data['role'].value_counts()
role_pcts = data['role'].value_counts(normalize=True) * 100
tenure_stats = data['years_in_company'].agg(['min', 'mean', 'max'])
hours_stats = data['hours_worked_per_week'].agg(['min', 'mean', 'max'])
salary_stats = data['monthly_salary_usd'].agg(['min', 'mean', 'max'])
burnout_counts = data['burnout_risk'].value_counts()
burnout_pcts = data['burnout_risk'].value_counts(normalize=True) * 100
work_mode_counts = data['work_mode'].value_counts()
work_mode_pcts = data['work_mode'].value_counts(normalize=True) * 100

section2_stats = stat_cards({
    "Total Employees": f"{len(data):,}",
    "Age Range": f"{age_stats['min']:.0f}–{age_stats['max']:.0f} yrs (avg {age_stats['mean']:.1f})",
    "Avg Tenure": f"{tenure_stats['mean']:.1f} yrs",
    "Avg Hours/Week": f"{hours_stats['mean']:.1f} hrs",
    "Avg Salary": f"${salary_stats['mean']:,.0f}",
})

def simple_dist_table(counts, pcts, col_label):
    rows = "".join(
        f"<tr><th>{idx}</th><td>{counts[idx]:,}</td><td>{pcts[idx]:.1f}%</td></tr>"
        for idx in counts.index
    )
    return f"""
    <div class="table-container">
    <table>
        <thead><tr><th>{col_label}</th><th>Employees</th><th>Percentage</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """

section2_html = f"""
<div class="section" id="workforce-overview">
    <h2>2. Workforce Overview</h2>
    <p class="section-desc">Who the workforce is: demographics, roles, tenure, and compensation.</p>
    {section2_stats}
    <div class="subsection">
        <h3>Gender Distribution</h3>
        {simple_dist_table(gender_counts, gender_pcts, "Gender")}
    </div>
    <div class="subsection">
        <h3>Role Distribution</h3>
        {simple_dist_table(role_counts, role_pcts, "Role")}
    </div>
    <div class="subsection">
        <h3>Work Mode Distribution</h3>
        {simple_dist_table(work_mode_counts, work_mode_pcts, "Work Mode")}
    </div>
    <div class="subsection">
        <h3>Overall Burnout Risk Distribution</h3>
        {simple_dist_table(burnout_counts, burnout_pcts, "Burnout Risk")}
    </div>
</div>
"""

print("Sections 1-2 computed.")

# =========================================================================
# SECTION 3 — Wellbeing & Workload Indicators
# =========================================================================
WELLBEING_SCORES = {
    "Work-Life Balance": "work_life_balance_score",
    "Job Satisfaction": "job_satisfaction_score",
    "Stress Level (self-report)": "stress_level_self_report",
    "Anxiety Score": "anxiety_score",
    "Depression Score": "depression_score",
    "Sleep Quality": "sleep_quality_score",
    "Social Support": "social_support_score",
    "Manager Support": "manager_support_score",
}
WORKLOAD_MEASURES = {
    "Commute Time (mins)": "commute_time_mins",
    "Absenteeism (days/month)": "absenteeism_days_per_month",
    "Overtime (hrs/week)": "overtime_hours_per_week",
    "Workload Index": "workload_index",
}

def minmaxmean_table(col_dict):
    rows = ""
    for label, col in col_dict.items():
        s = data[col].agg(['min', 'mean', 'max'])
        rows += f"<tr><th>{label}</th><td>{s['min']:.1f}</td><td>{s['mean']:.2f}</td><td>{s['max']:.1f}</td></tr>"
    return f"""
    <div class="table-container">
    <table>
        <thead><tr><th>Indicator</th><th>Min</th><th>Mean</th><th>Max</th></tr></thead>
        <tbody>{rows}</tbody>
    </table>
    </div>
    """

benefit_counts = data['mental_health_benefit_utilized'].value_counts()
benefit_pcts = data['mental_health_benefit_utilized'].value_counts(normalize=True) * 100
benefit_labels = {0: "Not Utilized", 1: "Utilized"}

benefit_rows = "".join(
    f"<tr><th>{benefit_labels.get(v, v)}</th><td>{benefit_counts[v]:,}</td><td>{benefit_pcts[v]:.1f}%</td></tr>"
    for v in benefit_counts.index
)
benefit_table_html = f"""
<div class="table-container">
<table>
    <thead><tr><th>Status</th><th>Employees</th><th>Percentage</th></tr></thead>
    <tbody>{benefit_rows}</tbody>
</table>
</div>
"""

# Wellbeing scores by benefit utilization (used vs not used)
compare_cols = {**WELLBEING_SCORES, "Workload Index": "workload_index"}
compare_rows = ""
for label, col in compare_cols.items():
    grouped = data.groupby('mental_health_benefit_utilized')[col].mean()
    not_used = grouped.get(0, float('nan'))
    used = grouped.get(1, float('nan'))
    diff = used - not_used
    sign = "+" if diff >= 0 else ""
    diff_cls = 'cell-high' if diff > 0 else 'cell-low'
    compare_rows += (
        f"<tr><th>{label}</th><td>{not_used:.2f}</td><td>{used:.2f}</td>"
        f"<td class='{diff_cls}'>{sign}{diff:.2f}</td></tr>"
    )
compare_table_html = f"""
<div class="table-container">
<table>
    <thead><tr><th>Indicator</th><th>Not Utilized</th><th>Utilized</th><th>Difference</th></tr></thead>
    <tbody>{compare_rows}</tbody>
</table>
</div>
"""

section3_html = f"""
<div class="section" id="wellbeing-workload">
    <h2>3. Wellbeing &amp; Workload Indicators</h2>
    <p class="section-desc">The 12 core indicators used throughout this report: scores, workload measures, and mental health benefit usage.</p>
    <div class="subsection">
        <h3>Wellbeing Scores — Overview</h3>
        {minmaxmean_table(WELLBEING_SCORES)}
    </div>
    <div class="subsection">
        <h3>Workload Measures — Overview</h3>
        {minmaxmean_table(WORKLOAD_MEASURES)}
    </div>
    <div class="subsection">
        <h3>Mental Health Benefit Utilization</h3>
        {benefit_table_html}
    </div>
    <div class="subsection">
        <h3>Wellbeing Scores by Benefit Utilization</h3>
        <p style="color:{COLOR_CREAM}; margin-left:4px; font-size:0.9em;">
            Employees who used the mental health benefit report higher stress/anxiety on average —
            consistent with people seeking support <i>because</i> they are struggling, not the benefit
            causing harm.
        </p>
        {compare_table_html}
    </div>
</div>
"""

print("Section 3 computed.")

# =========================================================================
# SECTION 4 — Linear Correlation with Burnout Risk
# =========================================================================
risk_map = {'Low': 0, 'Moderate': 1, 'High': 2}
data['burnout_risk_numeric'] = data['burnout_risk'].map(risk_map)

numeric_cols = data.select_dtypes(include='number').columns.drop(['employee_id', 'burnout_risk_numeric'])
correlations = data[numeric_cols].corrwith(data['burnout_risk_numeric']).sort_values(ascending=False)

CORR_LABELS = {
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

def strength_label(value: float) -> str:
    abs_v = abs(value)
    if abs_v >= 0.4:
        return "Strong"
    if abs_v >= 0.2:
        return "Moderate"
    if abs_v >= 0.1:
        return "Weak"
    return "Negligible"

corr_rows = ""
for col, value in correlations.items():
    label = CORR_LABELS.get(col, col)
    corr_rows += (
        f"<tr><th>{label}</th><td>{value:+.3f}</td>"
        f"<td>{strength_label(value)}</td></tr>"
    )

corr_table_html = f"""
<div class="table-container">
<table>
    <thead><tr><th>Factor</th><th>Correlation</th><th>Strength</th></tr></thead>
    <tbody>{corr_rows}</tbody>
</table>
</div>
"""

top_risk_factor = correlations.index[0]
top_protective_factor = correlations.index[-1]

section4_html = f"""
<div class="section" id="linear-correlation">
    <h2>4. Correlation with Burnout Risk (Linear)</h2>
    <p class="section-desc">Pearson correlation between every numeric factor and an ordered burnout_risk scale (Low=0, Moderate=1, High=2).</p>
    {corr_table_html}
    <div class="takeaway-panel">
        <div class="title">Key Takeaways</div>
        <ul>
            <li><b>Strongest Risk Factor:</b> {CORR_LABELS.get(top_risk_factor, top_risk_factor)} ({correlations[top_risk_factor]:+.3f})</li>
            <li><b>Strongest Protective Factor:</b> {CORR_LABELS.get(top_protective_factor, top_protective_factor)} ({correlations[top_protective_factor]:+.3f})</li>
            <li><b>Top 3 Risk Drivers:</b> {", ".join(CORR_LABELS.get(c, c) for c in correlations.head(3).index)}</li>
            <li><b>Top 3 Protective Factors:</b> {", ".join(CORR_LABELS.get(c, c) for c in correlations.tail(3).index)}</li>
        </ul>
    </div>
</div>
"""

print("Section 4 computed.")

# =========================================================================
# SECTION 5 — Nonlinear Relationship with Burnout Risk (Mutual Information)
# =========================================================================
mi_cols = list(numeric_cols)
X = data[mi_cols].fillna(data[mi_cols].median())
y = data['burnout_risk_numeric']

mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
mi_series = pd.Series(mi_scores, index=mi_cols).sort_values(ascending=False)

def mi_strength_label(value: float) -> str:
    if value >= 0.1:
        return "Weak"
    return "Negligible"

mi_rows = ""
for col, value in mi_series.items():
    label = CORR_LABELS.get(col, col)
    mi_rows += f"<tr><th>{label}</th><td>{value:.3f}</td><td>{mi_strength_label(value)}</td></tr>"

mi_table_html = f"""
<div class="table-container">
<table>
    <thead><tr><th>Factor</th><th>Mutual Information</th><th>Nonlinear Association</th></tr></thead>
    <tbody>{mi_rows}</tbody>
</table>
</div>
"""

section5_html = f"""
<div class="section" id="nonlinear-correlation">
    <h2>5. Nonlinear Relationship with Burnout Risk</h2>
    <p class="section-desc">Mutual Information scores capture non-linear associations that Pearson correlation can miss.</p>
    {mi_table_html}
    <div class="takeaway-panel">
        <div class="title">Key Takeaway</div>
        <ul>
            <li>No strong non-linear relationships were found beyond what linear correlation already showed.
            The top non-linear drivers (<b>{CORR_LABELS.get(mi_series.index[0], mi_series.index[0])}</b>,
            <b>{CORR_LABELS.get(mi_series.index[1], mi_series.index[1])}</b>,
            <b>{CORR_LABELS.get(mi_series.index[2], mi_series.index[2])}</b>) match the top linear drivers,
            meaning simple linear relationships explain most of the pattern.</li>
        </ul>
    </div>
</div>
"""

print("Section 5 computed.")

# =========================================================================
# SECTION 6 — Burnout Risk by Employee Profile (Role, Gender, Work Mode)
# =========================================================================
section6_html = f"""
<div class="section" id="profile-breakdown">
    <h2>6. Burnout Risk by Employee Profile</h2>
    <p class="section-desc">Burnout risk distribution across role, gender, and work mode.</p>
    <div class="subsection">
        <h3>By Role</h3>
        {df_to_distribution_table(data, 'role', 'Role')}
    </div>
    <div class="subsection">
        <h3>By Gender</h3>
        {df_to_distribution_table(data, 'gender', 'Gender')}
    </div>
    <div class="subsection">
        <h3>By Work Mode</h3>
        {df_to_distribution_table(data, 'work_mode', 'Work Mode')}
    </div>
</div>
"""

print("Section 6 computed.")

# =========================================================================
# SECTION 7 — Burnout Risk by Working Hours
# =========================================================================
hours_bins = [29, 39, 49, 59, 70]
hours_labels = ['30-39', '40-49', '50-59', '60+']
data['hours_worked_group'] = pd.cut(
    data['hours_worked_per_week'], bins=hours_bins, labels=hours_labels, right=True
)

hours_grouped = data.groupby('hours_worked_group', observed=False)
hours_counts = hours_grouped.size()
hours_avg = hours_grouped['hours_worked_per_week'].mean()
hours_pct = hours_grouped['burnout_risk'].value_counts(normalize=True).unstack().fillna(0) * 100

hours_rows = ""
for idx in hours_counts.index:
    row = f"<tr><th>{idx}</th><td>{hours_counts[idx]:,}</td><td>{hours_avg[idx]:.1f}</td>"
    for col in ['Low', 'Moderate', 'High']:
        val = hours_pct.loc[idx, col] if col in hours_pct.columns else 0
        cls = risk_cell_class(col)
        row += f'<td class="{cls}">{val:.1f}%</td>'
    row += "</tr>"
    hours_rows += row

hours_table_html = f"""
<div class="table-container">
<table>
    <thead><tr><th>Hours Group</th><th>Employees</th><th>Avg Hours</th><th>Low</th><th>Moderate</th><th>High</th></tr></thead>
    <tbody>{hours_rows}</tbody>
</table>
</div>
"""

section7_html = f"""
<div class="section" id="working-hours">
    <h2>7. Burnout Risk by Working Hours</h2>
    <p class="section-desc">Burnout risk distribution across weekly working-hour groups — one of the clearest patterns in this dataset.</p>
    {hours_table_html}
    <div class="takeaway-panel">
        <div class="title">Key Takeaway</div>
        <ul>
            <li>High-risk burnout jumps from <b>2.3%</b> at 30-39 hrs/week to <b>57.7%</b> at 60+ hrs/week —
            roughly a 25x increase, and the single clearest pattern in the whole dataset.</li>
        </ul>
    </div>
</div>
"""

print("Section 7 computed.")

# =========================================================================
# SECTION 8 — Burnout Risk by Tenure & Salary
# =========================================================================
tenure_bins = [-1, 2, 5, 8, 11, 14, 100]
tenure_labels = ['0–2 years', '3–5 years', '6–8 years', '9–11 years', '12–14 years', '15+ years']
data['years_in_company_group'] = pd.cut(
    data['years_in_company'], bins=tenure_bins, labels=tenure_labels
)

tenure_html = df_to_distribution_table(data, 'years_in_company_group', 'Tenure')

data['salary_group'] = pd.qcut(data['monthly_salary_usd'], q=5)
salary_html = df_to_distribution_table(data, 'salary_group', 'Salary Group')

section8_html = f"""
<div class="section" id="tenure-salary">
    <h2>8. Burnout Risk by Tenure &amp; Salary</h2>
    <p class="section-desc">Burnout risk distribution across employee tenure and monthly salary groups.</p>
    <div class="subsection">
        <h3>By Years in Company</h3>
        {tenure_html}
    </div>
    <div class="subsection">
        <h3>By Monthly Salary (Quintiles)</h3>
        {salary_html}
    </div>
</div>
"""

print("Section 8 computed.")

# =========================================================================
# SECTION 9 — Burnout Risk by Company Support (Turnover, Benefit Usage)
# =========================================================================
turnover_html = df_to_distribution_table(data, 'left_company_last_year', 'Left Last Year')
benefit_dist_html = df_to_distribution_table(data, 'mental_health_benefit_utilized', 'Benefit Used')

section9_html = f"""
<div class="section" id="company-support">
    <h2>9. Burnout Risk by Company Support</h2>
    <p class="section-desc">Burnout risk distribution by employee turnover and mental health benefit utilization.</p>
    <div class="subsection">
        <h3>By Employee Turnover (left_company_last_year)</h3>
        {turnover_html}
    </div>
    <div class="subsection">
        <h3>By Mental Health Benefit Usage</h3>
        {benefit_dist_html}
    </div>
</div>
"""

print("Section 9 computed.")

# =========================================================================
# SECTION 10 — Interaction Effects
# =========================================================================
overtime_bins = [-1, 5, 10, 15, 25]
overtime_labels = ['0-5', '6-10', '11-15', '16+']
data['overtime_group'] = pd.cut(
    data['overtime_hours_per_week'], bins=overtime_bins, labels=overtime_labels
)

workload_bins = [-1, 3, 5, 7, 11]
workload_labels = ['Low (0-3)', 'Mid-Low (3-5)', 'Mid-High (5-7)', 'High (7+)']
data['workload_group'] = pd.cut(
    data['workload_index'], bins=workload_bins, labels=workload_labels
)

support_bins = [0, 2, 3, 4, 6]
support_labels = ['Low', 'Mid', 'Good', 'High']
data['manager_support_group'] = pd.cut(
    data['manager_support_score'], bins=support_bins, labels=support_labels, right=False
)

role_hours_pivot = data.pivot_table(
    index='role', columns='hours_worked_group', values='burnout_risk',
    aggfunc=high_risk_pct, observed=False
)
mode_overtime_pivot = data.pivot_table(
    index='work_mode', columns='overtime_group', values='burnout_risk',
    aggfunc=high_risk_pct, observed=False
)
support_workload_pivot = data.pivot_table(
    index='manager_support_group', columns='workload_group', values='burnout_risk',
    aggfunc=high_risk_pct, observed=False
)
role_support_pivot = data.pivot_table(
    index='role', columns='manager_support_group', values='burnout_risk',
    aggfunc=high_risk_pct, observed=False
)

def worst_combo(pivot: pd.DataFrame, row_name: str, col_name: str) -> str:
    stacked = pivot.stack()
    worst_idx = stacked.idxmax()
    worst_val = stacked.max()
    return f"{row_name} = {worst_idx[0]}, {col_name} = {worst_idx[1]} ({worst_val:.1f}%)"

section10_html = f"""
<div class="section" id="interaction-effects">
    <h2>10. Interaction Effects</h2>
    <p class="section-desc">Single-factor averages can hide how much worse things get when risk factors stack together. These four combinations reveal where risk concentrates.</p>
    <div class="subsection">
        <h3>Role × Hours Worked/Week (% High Risk)</h3>
        {pivot_to_heat_table(role_hours_pivot, "Role")}
    </div>
    <div class="subsection">
        <h3>Work Mode × Overtime (% High Risk)</h3>
        {pivot_to_heat_table(mode_overtime_pivot, "Work Mode")}
    </div>
    <div class="subsection">
        <h3>Manager Support × Workload Index (% High Risk)</h3>
        {pivot_to_heat_table(support_workload_pivot, "Manager Support")}
    </div>
    <div class="subsection">
        <h3>Role × Manager Support (% High Risk)</h3>
        {pivot_to_heat_table(role_support_pivot, "Role")}
    </div>
    <div class="takeaway-panel">
        <div class="title">Highest-Risk Combinations Found</div>
        <ul>
            <li><b>Role × Hours:</b> {worst_combo(role_hours_pivot, 'Role', 'Hours')}</li>
            <li><b>Work Mode × Overtime:</b> {worst_combo(mode_overtime_pivot, 'Mode', 'Overtime')}</li>
            <li><b>Manager Support × Workload:</b> {worst_combo(support_workload_pivot, 'Support', 'Workload')}</li>
            <li><b>Role × Manager Support:</b> {worst_combo(role_support_pivot, 'Role', 'Support')}</li>
        </ul>
    </div>
</div>
"""

print("Section 10 computed.")

# =========================================================================
# SECTION 11 — Final Summary / Key Takeaways
# =========================================================================
section11_html = f"""
<div class="final-summary" id="final-summary">
    <h2>11. Final Summary — What Drives Burnout Here</h2>
    <div class="final-grid">
        <div class="final-card risk">
            <h4>🔴 Strongest Risk Drivers</h4>
            <p>Stress level, workload index, and hours worked per week show the strongest positive
            correlation with burnout risk. High-risk burnout jumps from 2.3% at 30-39 hrs/week to
            57.7% at 60+ hrs/week.</p>
        </div>
        <div class="final-card protective">
            <h4>🟢 Strongest Protective Factors</h4>
            <p>Work-life balance and sleep quality show the strongest negative correlation —
            higher scores on these consistently mean lower burnout risk.</p>
        </div>
        <div class="final-card risk">
            <h4>🔴 Worst Combination Found</h4>
            <p>Low manager support + high workload index together produce an 87.5% high-risk rate —
            far worse than either factor alone would suggest.</p>
        </div>
        <div class="final-card protective">
            <h4>🟢 Manager Support is a Real Buffer</h4>
            <p>Even under high workload, employees with high manager support have a meaningfully
            lower high-risk rate (70.7%) than those with low support (87.5%) — especially for
            Manager and Executive roles.</p>
        </div>
        <div class="final-card risk">
            <h4>🔴 Role Sensitivity Differs</h4>
            <p>At 60+ hrs/week, Managers reach 73.8% high-risk while Support staff stay at 29.4% —
            the same hours affect different roles very differently.</p>
        </div>
        <div class="final-card">
            <h4>⚪ No Major Non-Linear Surprises</h4>
            <p>Mutual information analysis confirms the top non-linear drivers match the top linear
            drivers — the relationships in this dataset are mostly straightforward and monotonic.</p>
        </div>
    </div>
</div>
"""

print("Section 11 computed. Assembling final HTML...")

# =========================================================================
# ASSEMBLE FINAL HTML
# =========================================================================
header_html = f"""
<div class="report-header">
    <h1>🧠 HR Burnout Risk Analysis — Master Report</h1>
</div>
"""

toc_html = """
<div class="toc">
    <h2>Contents</h2>
    <ol>
        <li><a href="#data-quality">Data Quality Summary</a></li>
        <li><a href="#workforce-overview">Workforce Overview</a></li>
        <li><a href="#wellbeing-workload">Wellbeing &amp; Workload Indicators</a></li>
        <li><a href="#linear-correlation">Correlation with Burnout Risk (Linear)</a></li>
        <li><a href="#nonlinear-correlation">Nonlinear Relationship with Burnout Risk</a></li>
        <li><a href="#profile-breakdown">Burnout Risk by Employee Profile</a></li>
        <li><a href="#working-hours">Burnout Risk by Working Hours</a></li>
        <li><a href="#tenure-salary">Burnout Risk by Tenure &amp; Salary</a></li>
        <li><a href="#company-support">Burnout Risk by Company Support</a></li>
        <li><a href="#interaction-effects">Interaction Effects</a></li>
        <li><a href="#final-summary">Final Summary</a></li>
    </ol>
</div>
"""

footer_html = """
<footer>
</footer>
"""

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HR Burnout Risk Analysis — Master Report</title>
<style>
{CSS}
</style>
</head>
<body>
{header_html}
{toc_html}
{section1_html}
{section2_html}
{section3_html}
{section4_html}
{section5_html}
{section6_html}
{section7_html}
{section8_html}
{section9_html}
{section10_html}
{section11_html}
{footer_html}
</body>
</html>
"""

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"\nSaved master report to {OUTPUT_PATH}")