"""HR Attrition Analytics — Streamlit Dashboard
Phase 2: Risk Table with filters, selection, and employee drill-down card.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# PATHS & PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT  = ROOT / "outputs"

# Orange / black / white colour palette
ORANGE     = "#F97316"
BLACK      = "#0F172A"
WHITE      = "#FFFFFF"
GREY       = "#6B7280"
GREY_LIGHT = "#F3F4F6"
RED        = "#DC2626"

RISK_COLORS: dict[str, str] = {"High": RED, "Medium": ORANGE, "Low": GREY}

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be the first Streamlit call in the module
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="HR Attrition Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS THEME
# ═══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    """Apply orange-black-white theme via a single CSS injection block."""
    st.markdown(
        f"""
        <style>
        /* ── Layout ── */
        .main .block-container {{
            padding: 1.5rem 2rem 2rem;
            max-width: 1440px;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {{
            background-color: {BLACK};
        }}
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] div {{
            color: {WHITE} !important;
        }}
        [data-testid="stSidebar"] hr {{
            border-color: #374151;
        }}

        /* ── KPI card ── */
        .kpi {{
            background: {WHITE};
            border-radius: 8px;
            padding: 1.1rem 1.3rem;
            border-left: 4px solid {ORANGE};
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .kpi .lbl {{
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {GREY};
        }}
        .kpi .val {{
            font-size: 1.9rem;
            font-weight: 700;
            color: {BLACK};
            line-height: 1.15;
            margin: 0.15rem 0;
        }}
        .kpi .sub {{
            font-size: 0.74rem;
            color: #9CA3AF;
        }}

        /* ── Section heading ── */
        .sh {{
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {BLACK};
            border-bottom: 2px solid {ORANGE};
            padding-bottom: 0.3rem;
            margin-bottom: 0.9rem;
            margin-top: 0.5rem;
        }}

        /* ── Hide only the footer branding ── */
        footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# REUSABLE UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def kpi(label: str, value: str, sub: str = "") -> None:
    """Render an orange-accented KPI card."""
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="kpi">'
        f'<div class="lbl">{label}</div>'
        f'<div class="val">{value}</div>'
        f'{sub_html}</div>',
        unsafe_allow_html=True,
    )


def sh(text: str) -> None:
    """Render an orange-underlined section heading."""
    st.markdown(f'<div class="sh">{text}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS  (cached — loaded once per session)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_master() -> pd.DataFrame:
    """
    Merge the 5 raw CSVs and join with pre-computed ML risk scores.
    Row order mirrors the notebook merge so positional indices align
    with employee_idx in risk_scored_employees.csv.
    """
    employee     = pd.read_csv(DATA / "employee_data.csv")
    compensation = pd.read_csv(DATA / "compensation_data.csv")
    survey       = pd.read_csv(DATA / "survey_results.csv")
    attrition    = pd.read_csv(DATA / "attrition_data.csv")
    market       = pd.read_csv(DATA / "market_benchmarks.csv")

    # Compa_Ratio: how each employee's base salary compares to market benchmark
    comp_ext = (
        compensation
        .merge(employee[["Employee_ID", "Role", "Work_Location"]], on="Employee_ID")
        .merge(
            market.rename(columns={"Location": "Work_Location"}),
            on=["Role", "Work_Location"],
            how="left",
        )
    )
    comp_ext["Compa_Ratio"] = comp_ext["Base_Salary"] / comp_ext["Benchmark_Salary"]

    # Binary attrition label
    attrition["Attrition_Target"] = (attrition["Attrition_Status"] == "Yes").astype(int)

    # Master merge — retains Employee_ID and Role for display and drill-down
    df = (
        employee
        .merge(
            survey[[
                "Employee_ID", "Job_Satisfaction", "Work_Life_Balance",
                "Management_Support", "Career_Development",
                "Engagement_Level", "Feedback_Comments",
            ]],
            on="Employee_ID",
        )
        .merge(
            comp_ext[[
                "Employee_ID", "Base_Salary", "Bonus",
                "Stock_Options", "Total_Compensation", "Compa_Ratio",
            ]],
            on="Employee_ID",
        )
        .merge(
            attrition[["Employee_ID", "Attrition_Status", "Attrition_Target"]],
            on="Employee_ID",
        )
        .reset_index(drop=True)
    )

    # Join ML risk scores — integer row index aligns with notebook's X.index
    risk = pd.read_csv(OUT / "risk_scored_employees.csv")
    df = df.join(
        risk.set_index("employee_idx")[[
            "attrition_probability", "risk_tier",
            "cluster_persona", "expected_attrition_cost",
        ]]
    )

    return df


@st.cache_data
def load_retention_matrix() -> pd.DataFrame:
    """Load the per-persona retention strategy matrix."""
    return pd.read_csv(OUT / "retention_strategy_matrix.csv")


@st.cache_data
def load_model_eval() -> pd.DataFrame:
    """Load the three-model evaluation scorecard."""
    return pd.read_csv(OUT / "model_evaluation.csv")


@st.cache_data
def load_winner_meta() -> dict:
    """Load winner model name and decision threshold."""
    with open(OUT / "winner_meta.json") as fh:
        return json.load(fh)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

# Shared Plotly layout defaults
_LAYOUT = dict(
    paper_bgcolor=WHITE,
    plot_bgcolor=WHITE,
    font=dict(color=BLACK, size=11),
)


def chart_risk_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing the proportion of workforce in each risk tier."""
    counts = (
        df["risk_tier"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )
    att_rate = df["Attrition_Target"].mean()

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.62,
        marker=dict(
            colors=[RISK_COLORS[t] for t in counts.index],
            line=dict(color=WHITE, width=2),
        ),
        textinfo="percent",
        insidetextfont=dict(color=WHITE, size=12),
        hovertemplate="%{label}: %{value:,} employees<extra></extra>",
    ))

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h", y=-0.06, x=0.5, xanchor="center",
            font=dict(size=11),
        ),
        annotations=[dict(
            text=f"{att_rate:.1%}",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=BLACK),
        )],
        margin=dict(t=10, b=30, l=10, r=10),
        height=290,
        **_LAYOUT,
    )
    return fig


def chart_dept_bar(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart: attrition rate per department, flagging above-average depts."""
    avg = df["Attrition_Target"].mean()
    dept = (
        df.groupby("Department")["Attrition_Target"]
        .mean()
        .reset_index(name="Rate")
        .sort_values("Rate")
    )
    colors = [RED if r > avg else GREY for r in dept["Rate"]]

    fig = go.Figure(go.Bar(
        x=dept["Rate"],
        y=dept["Department"],
        orientation="h",
        marker_color=colors,
        text=[f"{r:.1%}" for r in dept["Rate"]],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(
        x=avg,
        line_dash="dash", line_color=ORANGE, line_width=1.5,
        annotation_text=f"Avg {avg:.1%}",
        annotation_font_color=ORANGE,
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(tickformat=".0%", gridcolor=GREY_LIGHT, zeroline=False),
        yaxis=dict(showgrid=False),
        margin=dict(t=10, b=10, l=10, r=60),
        height=300,
        **_LAYOUT,
    )
    return fig


def chart_persona_bar(df: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: headcount (bars) and average attrition risk (line) per persona."""
    grp = (
        df.dropna(subset=["cluster_persona"])
        .groupby("cluster_persona")
        .agg(
            headcount=("Employee_ID", "count"),
            avg_risk=("attrition_probability", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk", ascending=False)
    )

    fig = go.Figure()

    # Headcount bars
    fig.add_trace(go.Bar(
        x=grp["cluster_persona"],
        y=grp["headcount"],
        name="Headcount",
        marker_color=ORANGE,
        text=grp["headcount"].map("{:,}".format),
        textposition="outside",
        hovertemplate="%{x}<br>Headcount: %{y:,}<extra></extra>",
        yaxis="y",
    ))

    # Avg risk line (secondary axis)
    fig.add_trace(go.Scatter(
        x=grp["cluster_persona"],
        y=grp["avg_risk"],
        mode="markers+lines",
        name="Avg Risk",
        marker=dict(color=RED, size=9),
        line=dict(color=RED, width=1.5, dash="dot"),
        hovertemplate="%{x}<br>Avg Risk: %{y:.1%}<extra></extra>",
        yaxis="y2",
    ))

    fig.update_layout(
        yaxis=dict(title="Headcount", gridcolor=GREY_LIGHT, zeroline=False),
        yaxis2=dict(
            title="Avg Attrition Risk",
            overlaying="y", side="right",
            tickformat=".0%", showgrid=False,
            range=[0, grp["avg_risk"].max() * 1.5],
        ),
        xaxis=dict(showgrid=False),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
        margin=dict(t=30, b=10, l=10, r=60),
        height=310,
        **_LAYOUT,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def chart_survey_radar(row: pd.Series) -> go.Figure:
    """Spider/radar chart comparing an employee's 5 survey scores to the team average."""
    dims   = ["Job_Satisfaction", "Work_Life_Balance", "Management_Support",
              "Career_Development", "Engagement_Level"]
    labels = ["Job Sat.", "Work-Life", "Mgmt Support", "Career Dev.", "Engagement"]

    # Close the polygon by repeating first element
    vals   = [row[d] for d in dims] + [row[dims[0]]]
    labels = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=labels, fill="toself", name="Employee",
        line=dict(color=ORANGE, width=2),
        fillcolor=f"rgba(249,115,22,0.15)",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 5], tickfont=dict(size=9), gridcolor="#E5E7EB"),
            angularaxis=dict(gridcolor="#E5E7EB"),
            bgcolor=WHITE,
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=30, r=30),
        height=280,
        **_LAYOUT,
    )
    return fig


def _risk_drivers(row: pd.Series, df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Build a ranked list of proxy risk drivers by comparing the employee's
    feature values to population medians. Returns (driver_label, direction_hint).
    """
    drivers: list[tuple[float, str, str]] = []  # (severity, label, note)

    survey_cols = {
        "Job_Satisfaction":    "Low job satisfaction",
        "Work_Life_Balance":   "Poor work-life balance",
        "Management_Support":  "Low management support",
        "Career_Development":  "Limited career development",
        "Engagement_Level":    "Low engagement",
    }
    for col, label in survey_cols.items():
        med = df[col].median()
        if row[col] < med:
            drivers.append((med - row[col], label, f"{row[col]:.1f} vs median {med:.1f}"))

    # Compensation vs market
    if pd.notna(row.get("Compa_Ratio")):
        if row["Compa_Ratio"] < 0.95:
            gap = (1 - row["Compa_Ratio"]) * 100
            drivers.append((gap, "Below-market pay", f"Compa-ratio {row['Compa_Ratio']:.2f}"))

    # Tenure risk band (very new <1yr or stagnant >8yr both correlate with attrition)
    if row["Tenure_Years"] < 1:
        drivers.append((0.5, "New hire (<1 yr tenure)", f"{row['Tenure_Years']:.1f} yrs"))
    elif row["Tenure_Years"] > 8:
        drivers.append((0.3, "Long-tenure stagnation (>8 yrs)", f"{row['Tenure_Years']:.1f} yrs"))

    # Sort by severity, return top 5
    drivers.sort(key=lambda x: x[0], reverse=True)
    return [(lbl, note) for _, lbl, note in drivers[:5]]


def page_overview(df: pd.DataFrame) -> None:
    """
    Overview page: top-level KPIs and key distribution charts.
    Designed to answer — how large is the risk, and where is it concentrated?
    """
    total      = len(df)
    att_rate   = df["Attrition_Target"].mean()
    high_risk  = int((df["risk_tier"] == "High").sum())
    total_cost = df["expected_attrition_cost"].sum()

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Total Employees", f"{total:,}")
    with c2:
        kpi("Historical Attrition", f"{att_rate:.1%}", "of labelled workforce")
    with c3:
        kpi("High-Risk Employees", f"{high_risk:,}", f"{high_risk / total:.1%} of workforce")
    with c4:
        kpi("Expected Attrition Cost", f"${total_cost / 1_000_000:.1f}M", "@ 1.5x annual salary replacement")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk tier distribution + department attrition ─────────────────────────
    col_l, col_r = st.columns([1, 2])
    with col_l:
        sh("Risk Tier Distribution")
        st.plotly_chart(
            chart_risk_donut(df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_r:
        sh("Attrition Rate by Department")
        st.plotly_chart(
            chart_dept_bar(df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Persona breakdown ────────────────────────────────────────────────────
    sh("Employee Personas — Headcount & Average Attrition Risk")
    st.plotly_chart(
        chart_persona_bar(df),
        use_container_width=True,
        config={"displayModeBar": False},
    )


def page_risk_table(df: pd.DataFrame) -> None:
    """
    Filterable risk table with an employee drill-down panel.
    Filters live in the sidebar (injected contextually).
    Selecting a row opens a detailed card below the table.
    """
    # ── Sidebar filters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Filters**")

        tier_opts   = ["High", "Medium", "Low"]
        sel_tiers   = st.multiselect("Risk Tier", tier_opts, default=["High", "Medium"])

        dept_opts   = sorted(df["Department"].unique())
        sel_depts   = st.multiselect("Department", dept_opts, default=dept_opts)

        persona_opts = sorted(df["cluster_persona"].dropna().unique())
        sel_personas = st.multiselect("Persona", persona_opts, default=persona_opts)

        min_prob, max_prob = st.slider(
            "Attrition Probability (%)",
            min_value=0, max_value=100, value=(0, 100), step=1,
        )

    # ── Apply filters ─────────────────────────────────────────────────────────
    mask = (
        df["risk_tier"].isin(sel_tiers if sel_tiers else tier_opts)
        & df["Department"].isin(sel_depts if sel_depts else dept_opts)
        & df["cluster_persona"].isin(sel_personas if sel_personas else persona_opts)
        & df["attrition_probability"].between(min_prob / 100, max_prob / 100)
    )
    view = df.loc[mask].copy()

    # ── Filter summary ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi("Matching Employees", f"{len(view):,}", f"of {len(df):,} total")
    with c2:
        kpi("Avg Attrition Risk",
            f"{view['attrition_probability'].mean():.1%}" if len(view) else "—",
            "within filter selection")
    with c3:
        total_cost = view["expected_attrition_cost"].sum()
        kpi("Total Expected Cost", f"${total_cost / 1_000_000:.1f}M",
            "replacement cost estimate")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build display table ───────────────────────────────────────────────────
    display_cols = {
        "Employee_ID":            "ID",
        "Department":             "Department",
        "Role":                   "Role",
        "Age":                    "Age",
        "Tenure_Years":           "Tenure",
        "Employment_Type":        "Type",
        "Work_Location":          "Location",
        "cluster_persona":        "Persona",
        "risk_tier":              "Risk",
        "attrition_probability":  "Prob %",
        "expected_attrition_cost":"Est. Cost",
    }
    tbl = view[list(display_cols)].rename(columns=display_cols).reset_index(drop=True)

    sh("Employee Risk Table  —  select a row to open the drill-down card")

    # st.dataframe with row selection (Streamlit >= 1.31 feature)
    event = st.dataframe(
        tbl,
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={
            "Prob %": st.column_config.ProgressColumn(
                "Prob %", format="%.1%", min_value=0, max_value=1,
            ),
            "Est. Cost": st.column_config.NumberColumn(
                "Est. Cost", format="$%,.0f",
            ),
            "Risk": st.column_config.TextColumn("Risk"),
        },
        on_select="rerun",
        selection_mode="single-row",
    )

    # ── Drill-down card ───────────────────────────────────────────────────────
    selected_rows = event.selection.rows if event and event.selection else []
    if not selected_rows:
        st.caption("Select a row above to view the employee drill-down card.")
        return

    row_idx   = selected_rows[0]
    row_id    = tbl.iloc[row_idx]["ID"]
    emp_row   = df.loc[df["Employee_ID"] == row_id].iloc[0]

    st.markdown("<br>", unsafe_allow_html=True)
    sh(f"Drill-down  —  {emp_row['Employee_ID']}  |  {emp_row['Role']}  |  {emp_row['Department']}")

    left, mid, right = st.columns([1, 1, 1])

    # Left column — identity & compensation
    with left:
        tier  = emp_row["risk_tier"]
        color = RISK_COLORS.get(tier, GREY)
        st.markdown(
            f"<div style='display:inline-block;background:{color};color:{WHITE};"
            f"border-radius:4px;padding:2px 10px;font-weight:700;font-size:0.82rem;"
            f"margin-bottom:0.6rem'>{tier} Risk</div>",
            unsafe_allow_html=True,
        )
        fields = [
            ("Age",              f"{emp_row['Age']} yrs"),
            ("Tenure",           f"{emp_row['Tenure_Years']:.1f} yrs"),
            ("Employment",       emp_row["Employment_Type"]),
            ("Location",         emp_row["Work_Location"]),
            ("Education",        emp_row["Education_Level"]),
            ("Marital Status",   emp_row["Marital_Status"]),
            ("Gender",           emp_row["Gender"]),
            ("Persona",          str(emp_row.get("cluster_persona", "—"))),
            ("Attrition Status", emp_row.get("Attrition_Status", "—")),
        ]
        rows_html = "".join(
            f"<tr>"
            f"<td style='padding:4px 8px 4px 0;color:{GREY};font-size:0.78rem;'>{k}</td>"
            f"<td style='padding:4px 0;font-size:0.82rem;font-weight:600;color:{BLACK};'>{v}</td>"
            f"</tr>"
            for k, v in fields
        )
        st.markdown(
            f"<table style='border-collapse:collapse;width:100%'>{rows_html}</table>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        comp_ratio = emp_row.get("Compa_Ratio", None)
        comp_label = f"{comp_ratio:.2f}" if pd.notna(comp_ratio) else "—"
        st.markdown(
            f"<div style='font-size:0.75rem;color:{GREY};margin-bottom:2px'>BASE SALARY</div>"
            f"<div style='font-size:1.5rem;font-weight:700;color:{BLACK}'>"
            f"${emp_row['Base_Salary']:,.0f}</div>"
            f"<div style='font-size:0.75rem;color:{GREY}'>Compa-ratio: <b>{comp_label}</b></div>",
            unsafe_allow_html=True,
        )

    # Middle column — survey radar
    with mid:
        st.markdown(
            f"<div style='font-size:0.78rem;font-weight:700;color:{GREY};"
            f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem'>"
            "Survey Profile</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            chart_survey_radar(emp_row),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        prob = emp_row["attrition_probability"]
        cost = emp_row["expected_attrition_cost"]
        st.markdown(
            f"<div style='text-align:center;margin-top:-0.5rem'>"
            f"<span style='font-size:1.6rem;font-weight:700;color:{RISK_COLORS.get(tier, GREY)}'>"
            f"{prob:.1%}</span>"
            f"<span style='font-size:0.78rem;color:{GREY};margin-left:6px'>"
            f"attrition probability</span><br>"
            f"<span style='font-size:0.85rem;color:{BLACK};font-weight:600'>"
            f"${cost:,.0f}</span>"
            f"<span style='font-size:0.75rem;color:{GREY};margin-left:4px'>"
            f"replacement cost est.</span></div>",
            unsafe_allow_html=True,
        )

    # Right column — proxy risk drivers + feedback
    with right:
        st.markdown(
            f"<div style='font-size:0.78rem;font-weight:700;color:{GREY};"
            f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.5rem'>"
            "Key Risk Signals</div>",
            unsafe_allow_html=True,
        )
        drivers = _risk_drivers(emp_row, df)
        if drivers:
            for lbl, note in drivers:
                st.markdown(
                    f"<div style='border-left:3px solid {ORANGE};padding:5px 10px;"
                    f"margin-bottom:6px;background:{GREY_LIGHT};border-radius:0 4px 4px 0'>"
                    f"<div style='font-size:0.82rem;font-weight:600;color:{BLACK}'>{lbl}</div>"
                    f"<div style='font-size:0.74rem;color:{GREY}'>{note}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f"<div style='color:{GREY};font-size:0.82rem'>No significant risk signals detected.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        feedback = str(emp_row.get("Feedback_Comments", "")).strip()
        if feedback and feedback.lower() not in ("nan", ""):
            st.markdown(
                f"<div style='font-size:0.78rem;font-weight:700;color:{GREY};"
                f"text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.4rem'>"
                "Feedback Comment</div>"
                f"<div style='font-style:italic;font-size:0.85rem;color:{BLACK};"
                f"background:{GREY_LIGHT};padding:10px 12px;border-radius:6px;"
                f"border-left:3px solid {GREY}'>\"{feedback}\"</div>",
                unsafe_allow_html=True,
            )


def page_analytics(df: pd.DataFrame) -> None:
    """
    Analytics Explorer: four tabbed sections covering workforce composition,
    tenure & attrition, compensation signals, and survey heat-maps.
    """
    tab_comp, tab_tenure, tab_comp_pay, tab_survey = st.tabs([
        "Workforce Composition",
        "Tenure & Attrition",
        "Compensation Signals",
        "Survey Heat-map",
    ])

    # ── Tab 1 — Workforce Composition ─────────────────────────────────────────
    with tab_comp:
        sh("Workforce Composition")
        col_a, col_b = st.columns(2)

        # Dept × risk tier stacked bar
        with col_a:
            grp = (
                df.groupby(["Department", "risk_tier"])
                .size()
                .reset_index(name="n")
            )
            fig = go.Figure()
            for tier, color in [("High", RED), ("Medium", ORANGE), ("Low", GREY)]:
                sub = grp[grp["risk_tier"] == tier]
                fig.add_trace(go.Bar(
                    x=sub["Department"], y=sub["n"],
                    name=tier, marker_color=color,
                    hovertemplate="%{x}  |  " + tier + ": %{y:,}<extra></extra>",
                ))
            fig.update_layout(
                barmode="stack", title="Risk Tiers by Department",
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor=GREY_LIGHT),
                legend=dict(orientation="h", y=1.06),
                margin=dict(t=40, b=10, l=10, r=10), height=320,
                **_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Employment type donut
        with col_b:
            counts = df["Employment_Type"].value_counts()
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values, hole=0.55,
                marker=dict(
                    colors=[ORANGE, BLACK, GREY],
                    line=dict(color=WHITE, width=2),
                ),
                textinfo="percent+label",
                hovertemplate="%{label}: %{value:,}<extra></extra>",
            ))
            fig.update_layout(
                title="Employment Type Mix",
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10), height=320,
                **_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Location bar
        loc = (
            df.groupby("Work_Location")["Attrition_Target"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "attrited", "count": "total"})
            .assign(rate=lambda x: x["attrited"] / x["total"])
            .sort_values("rate", ascending=False)
            .reset_index()
        )
        fig = go.Figure(go.Bar(
            x=loc["Work_Location"], y=loc["rate"],
            marker_color=[RED if r > df["Attrition_Target"].mean() else GREY for r in loc["rate"]],
            text=[f"{r:.1%}" for r in loc["rate"]],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        ))
        avg = df["Attrition_Target"].mean()
        fig.add_hline(y=avg, line_dash="dash", line_color=ORANGE, line_width=1.5,
                      annotation_text=f"Avg {avg:.1%}", annotation_font_color=ORANGE)
        fig.update_layout(
            title="Attrition Rate by Work Location",
            xaxis=dict(showgrid=False),
            yaxis=dict(tickformat=".0%", gridcolor=GREY_LIGHT),
            margin=dict(t=40, b=10, l=10, r=10), height=300,
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 2 — Tenure & Attrition ────────────────────────────────────────────
    with tab_tenure:
        sh("Tenure & Attrition Patterns")
        col_a, col_b = st.columns(2)

        # Tenure histogram split by attrition status
        with col_a:
            fig = go.Figure()
            for label, color, val in [("Stayed", GREY, 0), ("Left", RED, 1)]:
                subset = df[df["Attrition_Target"] == val]["Tenure_Years"]
                fig.add_trace(go.Histogram(
                    x=subset, name=label, marker_color=color,
                    opacity=0.75, nbinsx=20,
                    hovertemplate="Tenure %{x:.1f} yrs: %{y:,}<extra></extra>",
                ))
            fig.update_layout(
                title="Tenure Distribution (Stayed vs Left)",
                barmode="overlay",
                xaxis=dict(title="Tenure (years)", gridcolor=GREY_LIGHT),
                yaxis=dict(title="Headcount", gridcolor=GREY_LIGHT),
                legend=dict(orientation="h", y=1.06),
                margin=dict(t=40, b=10, l=10, r=10), height=320,
                **_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Age vs attrition probability scatter
        with col_b:
            samp = df.sample(min(2000, len(df)), random_state=42)
            fig = go.Figure(go.Scatter(
                x=samp["Age"], y=samp["attrition_probability"],
                mode="markers",
                marker=dict(
                    color=samp["attrition_probability"],
                    colorscale=[[0, GREY], [0.5, ORANGE], [1, RED]],
                    size=5, opacity=0.6,
                    showscale=True,
                    colorbar=dict(title="Risk", tickformat=".0%"),
                ),
                hovertemplate="Age %{x}  |  Risk %{y:.1%}<extra></extra>",
            ))
            fig.update_layout(
                title="Age vs Attrition Probability (sample n=2,000)",
                xaxis=dict(title="Age", gridcolor=GREY_LIGHT),
                yaxis=dict(title="Attrition Probability", tickformat=".0%", gridcolor=GREY_LIGHT),
                margin=dict(t=40, b=10, l=10, r=10), height=320,
                **_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Tenure band attrition rate bar
        df_t = df.copy()
        df_t["Tenure_Band"] = pd.cut(
            df_t["Tenure_Years"],
            bins=[0, 1, 2, 4, 6, 10, 99],
            labels=["<1 yr", "1-2 yr", "2-4 yr", "4-6 yr", "6-10 yr", "10+ yr"],
        )
        band = (
            df_t.groupby("Tenure_Band", observed=True)["Attrition_Target"]
            .mean()
            .reset_index(name="Rate")
        )
        fig = go.Figure(go.Bar(
            x=band["Tenure_Band"].astype(str), y=band["Rate"],
            marker_color=ORANGE,
            text=[f"{r:.1%}" for r in band["Rate"]],
            textposition="outside",
            hovertemplate="%{x}: %{y:.1%}<extra></extra>",
        ))
        fig.update_layout(
            title="Attrition Rate by Tenure Band",
            xaxis=dict(showgrid=False),
            yaxis=dict(tickformat=".0%", gridcolor=GREY_LIGHT),
            margin=dict(t=40, b=10, l=10, r=10), height=290,
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 3 — Compensation Signals ──────────────────────────────────────────
    with tab_comp_pay:
        sh("Compensation Signals")
        col_a, col_b = st.columns(2)

        # Compa-ratio box by risk tier
        with col_a:
            fig = go.Figure()
            for tier, color in [("High", RED), ("Medium", ORANGE), ("Low", GREY)]:
                vals = df.loc[df["risk_tier"] == tier, "Compa_Ratio"].dropna()
                fig.add_trace(go.Box(
                    y=vals, name=tier, marker_color=color,
                    line=dict(color=color), boxmean=True,
                    hovertemplate="Compa-ratio %{y:.2f}<extra></extra>",
                ))
            fig.add_hline(y=1.0, line_dash="dash", line_color=BLACK, line_width=1,
                          annotation_text="Market parity", annotation_font_color=BLACK)
            fig.update_layout(
                title="Compa-Ratio Distribution by Risk Tier",
                yaxis=dict(title="Compa-Ratio", gridcolor=GREY_LIGHT),
                xaxis=dict(showgrid=False),
                legend=dict(orientation="h", y=1.06),
                margin=dict(t=40, b=10, l=10, r=10), height=320,
                **_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Salary vs attrition probability by dept
        with col_b:
            dept_pay = (
                df.groupby("Department")
                .agg(avg_salary=("Base_Salary", "mean"),
                     avg_risk=("attrition_probability", "mean"),
                     headcount=("Employee_ID", "count"))
                .reset_index()
            )
            fig = go.Figure(go.Scatter(
                x=dept_pay["avg_salary"],
                y=dept_pay["avg_risk"],
                mode="markers+text",
                text=dept_pay["Department"],
                textposition="top center",
                textfont=dict(size=10),
                marker=dict(
                    size=dept_pay["headcount"] / dept_pay["headcount"].max() * 40 + 10,
                    color=ORANGE, opacity=0.8,
                    line=dict(color=BLACK, width=1),
                ),
                hovertemplate="%{text}<br>Avg Salary: $%{x:,.0f}<br>Avg Risk: %{y:.1%}<extra></extra>",
            ))
            fig.update_layout(
                title="Dept: Avg Salary vs Avg Attrition Risk (bubble = headcount)",
                xaxis=dict(title="Avg Base Salary ($)", gridcolor=GREY_LIGHT,
                           tickformat="$,.0f"),
                yaxis=dict(title="Avg Attrition Probability", tickformat=".0%",
                           gridcolor=GREY_LIGHT),
                margin=dict(t=40, b=30, l=10, r=10), height=320,
                **_LAYOUT,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Compa-ratio vs attrition probability scatter
        samp = df.dropna(subset=["Compa_Ratio"]).sample(min(2000, len(df)), random_state=7)
        fig = go.Figure(go.Scatter(
            x=samp["Compa_Ratio"], y=samp["attrition_probability"],
            mode="markers",
            marker=dict(
                color=samp["attrition_probability"],
                colorscale=[[0, GREY], [0.5, ORANGE], [1, RED]],
                size=5, opacity=0.55,
                showscale=True,
                colorbar=dict(title="Risk", tickformat=".0%"),
            ),
            hovertemplate="Compa-ratio %{x:.2f}  |  Risk %{y:.1%}<extra></extra>",
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color=BLACK, line_width=1,
                      annotation_text="Market parity", annotation_font_color=BLACK)
        fig.update_layout(
            title="Compa-Ratio vs Attrition Probability (sample n=2,000)",
            xaxis=dict(title="Compa-Ratio", gridcolor=GREY_LIGHT),
            yaxis=dict(title="Attrition Probability", tickformat=".0%", gridcolor=GREY_LIGHT),
            margin=dict(t=40, b=10, l=10, r=10), height=300,
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Tab 4 — Survey Heat-map ───────────────────────────────────────────────
    with tab_survey:
        sh("Survey Signal Heat-map")

        survey_dims = [
            "Job_Satisfaction", "Work_Life_Balance",
            "Management_Support", "Career_Development", "Engagement_Level",
        ]
        dim_labels = [
            "Job Sat.", "Work-Life", "Mgmt Support", "Career Dev.", "Engagement",
        ]

        group_by = st.selectbox(
            "Group by",
            ["Department", "cluster_persona", "risk_tier", "Work_Location"],
            format_func=lambda x: {
                "cluster_persona": "Persona",
                "risk_tier": "Risk Tier",
                "Work_Location": "Work Location",
            }.get(x, x),
        )

        heat = (
            df.groupby(group_by)[survey_dims]
            .mean()
            .round(2)
        )
        # Order rows by overall avg risk if possible
        if "risk_tier" in df.columns and group_by != "risk_tier":
            order = (
                df.groupby(group_by)["attrition_probability"]
                .mean()
                .sort_values(ascending=False)
                .index
            )
            heat = heat.reindex(order)

        fig = go.Figure(go.Heatmap(
            z=heat.values,
            x=dim_labels,
            y=heat.index.astype(str),
            colorscale=[[0, RED], [0.5, ORANGE], [1, "#22C55E"]],
            zmin=1, zmax=5,
            text=heat.values.round(2),
            texttemplate="%{text:.1f}",
            textfont=dict(size=11),
            hovertemplate="%{y}  |  %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Score", tickvals=[1, 2, 3, 4, 5]),
        ))
        fig.update_layout(
            title=f"Average Survey Scores by {group_by.replace('_', ' ')}",
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            margin=dict(t=50, b=10, l=10, r=10),
            height=max(280, len(heat) * 50 + 80),
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Survey score distributions for each dimension
        st.markdown("<br>", unsafe_allow_html=True)
        sh("Score Distributions Across Workforce")
        fig = go.Figure()
        for dim, lbl in zip(survey_dims, dim_labels):
            fig.add_trace(go.Violin(
                x=[lbl] * len(df), y=df[dim],
                name=lbl, box_visible=True, meanline_visible=True,
                fillcolor=ORANGE, opacity=0.6,
                line_color=BLACK, points=False,
                hovertemplate=lbl + ": %{y:.2f}<extra></extra>",
            ))
        fig.update_layout(
            violinmode="overlay",
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Score (1–5)", gridcolor=GREY_LIGHT, range=[0.5, 5.5]),
            showlegend=False,
            margin=dict(t=20, b=10, l=10, r=10), height=310,
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_retention_planner(df: pd.DataFrame, matrix: pd.DataFrame) -> None:
    """
    Retention Planner: persona strategy cards, at-risk employee breakdown,
    and an interactive savings calculator.
    """
    # ── Priority colour map ───────────────────────────────────────────────────
    PRIORITY_COLORS = {"High": RED, "Medium": ORANGE, "Low": GREY}

    # Pre-compute per-persona at-risk headcount from the live df
    persona_risk = (
        df[df["risk_tier"] == "High"]
        .groupby("cluster_persona")
        .agg(
            high_risk_n=("Employee_ID", "count"),
            total_cost=("expected_attrition_cost", "sum"),
        )
        .reset_index()
    )
    matrix_aug = matrix.merge(
        persona_risk, left_on="Persona", right_on="cluster_persona", how="left"
    ).fillna({"high_risk_n": 0, "total_cost": 0})

    # ── Persona strategy cards ────────────────────────────────────────────────
    sh("Persona Strategy Cards")
    cols = st.columns(len(matrix_aug))
    for col, (_, row) in zip(cols, matrix_aug.iterrows()):
        p_color = PRIORITY_COLORS.get(str(row["Priority"]).strip(), GREY)
        with col:
            st.markdown(
                f"<div style='background:{WHITE};border-radius:10px;"
                f"border-top:4px solid {p_color};"
                f"box-shadow:0 2px 6px rgba(0,0,0,0.08);"
                f"padding:1rem 1.1rem;height:100%'>"
                f"<div style='font-size:0.7rem;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:0.08em;color:{p_color};margin-bottom:0.3rem'>"
                f"{row['Priority']} Priority</div>"
                f"<div style='font-size:1rem;font-weight:700;color:{BLACK};"
                f"margin-bottom:0.6rem;line-height:1.3'>{row['Persona']}</div>"
                f"<div style='font-size:0.8rem;color:{GREY};margin-bottom:0.6rem'>"
                f"<b>{int(row['Headcount']):,}</b> employees &nbsp;|&nbsp; "
                f"<b style='color:{p_color}'>{int(row['high_risk_n']):,}</b> high-risk</div>"
                f"<div style='font-size:0.75rem;font-weight:700;color:{GREY};"
                f"text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem'>"
                f"Primary Driver</div>"
                f"<div style='font-size:0.82rem;color:{BLACK};margin-bottom:0.7rem'>"
                f"{row['Primary Driver']}</div>"
                f"<div style='font-size:0.75rem;font-weight:700;color:{GREY};"
                f"text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem'>"
                f"Recommended Action</div>"
                f"<div style='font-size:0.82rem;color:{BLACK};margin-bottom:0.7rem'>"
                f"{row['Recommended Action']}</div>"
                f"<div style='font-size:0.8rem;color:{GREY}'>"
                f"Est. Cost at Risk: "
                f"<b style='color:{BLACK}'>${row['total_cost']/1_000_000:.1f}M</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk breakdown by persona chart ──────────────────────────────────────
    sh("High-Risk Headcount by Persona")
    fig = go.Figure()
    ma = matrix_aug.sort_values("high_risk_n", ascending=False)
    pct_high = (ma["high_risk_n"] / ma["Headcount"] * 100).round(1)
    fig.add_trace(go.Bar(
        x=ma["Persona"], y=ma["Headcount"],
        name="Total Headcount", marker_color=GREY_LIGHT,
        hovertemplate="%{x}<br>Total: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=ma["Persona"], y=ma["high_risk_n"],
        name="High Risk", marker_color=RED,
        text=[f"{p}%" for p in pct_high],
        textposition="outside",
        hovertemplate="%{x}<br>High Risk: %{y:,} (%{text})<extra></extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Headcount", gridcolor=GREY_LIGHT),
        legend=dict(orientation="h", y=1.06),
        margin=dict(t=30, b=10, l=10, r=10), height=310,
        **_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Savings calculator ────────────────────────────────────────────────────
    sh("Retention Investment Savings Calculator")
    st.markdown(
        f"<div style='font-size:0.83rem;color:{GREY};margin-bottom:1rem'>"
        "Estimate the net saving from reducing attrition in a target population. "
        "Replacement cost assumes 1.5x annual salary. Intervention cost is set per employee."
        "</div>",
        unsafe_allow_html=True,
    )
    calc_left, calc_right = st.columns([1, 1])

    with calc_left:
        persona_opts = ["All personas"] + sorted(df["cluster_persona"].dropna().unique().tolist())
        sel_persona = st.selectbox("Target Persona", persona_opts)

        tier_opts_c = ["High only", "High + Medium", "All"]
        sel_tier_c  = st.selectbox("Target Risk Tier", tier_opts_c)

        target = df.copy()
        if sel_persona != "All personas":
            target = target[target["cluster_persona"] == sel_persona]
        if sel_tier_c == "High only":
            target = target[target["risk_tier"] == "High"]
        elif sel_tier_c == "High + Medium":
            target = target[target["risk_tier"].isin(["High", "Medium"])]

        n_target = len(target)
        baseline_cost = target["expected_attrition_cost"].sum()

        reduction_pct = st.slider(
            "Attrition Reduction (%)",
            min_value=5, max_value=80, value=30, step=5,
            help="Expected % reduction in attrition probability after intervention.",
        )
        cost_per_emp = st.number_input(
            "Intervention Cost per Employee ($)",
            min_value=500, max_value=50_000, value=3_000, step=500,
        )

    with calc_right:
        saving      = baseline_cost * (reduction_pct / 100)
        prog_cost   = n_target * cost_per_emp
        net_saving  = saving - prog_cost
        roi_pct     = (net_saving / prog_cost * 100) if prog_cost > 0 else 0
        net_color   = "#22C55E" if net_saving > 0 else RED

        st.markdown("<br>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        with r1:
            kpi("Target Employees", f"{n_target:,}",
                f"{sel_persona}  |  {sel_tier_c}")
        with r2:
            kpi("Baseline Attrition Cost", f"${baseline_cost/1_000_000:.2f}M",
                "pre-intervention expected replacement")
        st.markdown("<br>", unsafe_allow_html=True)
        r3, r4 = st.columns(2)
        with r3:
            kpi("Programme Cost", f"${prog_cost/1_000_000:.2f}M",
                f"{n_target:,} emp x ${cost_per_emp:,}")
        with r4:
            st.markdown(
                f"<div class='kpi' style='border-left-color:{net_color}'>"
                f"<div class='lbl'>Net Saving</div>"
                f"<div class='val' style='color:{net_color}'>"
                f"{'+ ' if net_saving>=0 else '- '}${abs(net_saving)/1_000_000:.2f}M "
                f"</div>"
                f"<div class='sub'>ROI {roi_pct:+.0f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # Waterfall chart
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Baseline Cost", "Saving", "Programme Cost", "Net Saving"],
            y=[baseline_cost, -saving, prog_cost, 0],
            connector=dict(line=dict(color=GREY, width=1)),
            decreasing=dict(marker_color="#22C55E"),
            increasing=dict(marker_color=RED),
            totals=dict(marker_color=net_color),
            text=[
                f"${baseline_cost/1e6:.2f}M",
                f"-${saving/1e6:.2f}M",
                f"+${prog_cost/1e6:.2f}M",
                f"${net_saving/1e6:.2f}M",
            ],
            textposition="outside",
        ))
        fig.update_layout(
            yaxis=dict(tickformat="$,.0f", gridcolor=GREY_LIGHT),
            xaxis=dict(showgrid=False),
            margin=dict(t=20, b=10, l=10, r=10), height=280,
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# Ordinal encoding to match the notebook's preprocessing exactly
_EDU_MAP = {"High School": 1, "Associate": 2, "Bachelor": 3, "Master": 4, "PhD": 5}

# OHE column sets the model was trained on
_DEPT_CATS  = ["Customer Support","Engineering","Finance","HR",
               "Marketing","Operations","Product","Sales"]
_TYPE_CATS  = ["Contract","Full-Time","Part-Time"]
_LOC_CATS   = ["Atlanta","Austin","Boston","Chicago","Denver",
               "New York","Remote","San Francisco","Seattle"]

MODEL_FEATURES = [
    "Tenure_Years","Education_Level",
    "Job_Satisfaction","Work_Life_Balance","Management_Support",
    "Career_Development","Engagement_Level",
    "Base_Salary","Bonus","Stock_Options","Total_Compensation",
    "Compa_Ratio","Feedback_Sentiment",
] + [f"Department_{c}" for c in _DEPT_CATS] \
  + [f"Employment_Type_{c}" for c in _TYPE_CATS] \
  + [f"Work_Location_{c}" for c in _LOC_CATS]


def _build_feature_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Replicate the notebook's feature engineering on uploaded data:
    ordinal-encode Education_Level, OHE Department/Employment_Type/Work_Location,
    set Feedback_Sentiment=0 (NLP skipped for interactive use).
    Returns a DataFrame with exactly MODEL_FEATURES columns in order.
    """
    X = raw.copy()

    # Ordinal
    X["Education_Level"] = X["Education_Level"].map(_EDU_MAP).fillna(3)  # default: Bachelor

    # OHE — create all expected dummy columns, fill missing with 0
    for dept in _DEPT_CATS:
        X[f"Department_{dept}"] = (X["Department"] == dept).astype(int)
    for t in _TYPE_CATS:
        X[f"Employment_Type_{t}"] = (X["Employment_Type"] == t).astype(int)
    for loc in _LOC_CATS:
        X[f"Work_Location_{loc}"] = (X["Work_Location"] == loc).astype(int)

    # NLP placeholder
    X["Feedback_Sentiment"] = 0.0

    return X[MODEL_FEATURES]


def _assign_tier(prob: float, threshold: float) -> str:
    """Map probability to High/Medium/Low using the trained threshold."""
    if prob >= threshold:
        return "High"
    if prob >= threshold * 0.55:
        return "Medium"
    return "Low"


def page_score_new_data(winner: dict) -> None:
    """
    Score New Data: interactive form to score a single employee.
    Results accumulate in session state and can be downloaded as CSV.
    """
    import joblib

    threshold = winner["threshold"]

    # Load market benchmarks once for Compa_Ratio auto-calculation
    market = pd.read_csv(DATA / "market_benchmarks.csv")
    market_lookup = market.set_index(["Role", "Location"])["Benchmark_Salary"].to_dict()

    # Session state list for accumulated scored employees
    if "scored_employees" not in st.session_state:
        st.session_state.scored_employees = []

    st.markdown(
        f"<div style='font-size:0.83rem;color:{GREY};margin-bottom:1rem'>"
        f"Fill in the employee details below and click <b>Score Employee</b>. "
        f"Results accumulate in the table below and can be downloaded as CSV. "
        f"Model: {winner['winner_model']} &nbsp;|&nbsp; Threshold: <b>{threshold:.4f}</b>."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Entry form ───────────────────────────────────────────────────────────
    sh("Employee Details")
    with st.form("score_form", clear_on_submit=True):
        id_col, dept_col, role_col = st.columns(3)
        with id_col:
            emp_id = st.text_input("Employee ID", value="EMP-NEW")
        with dept_col:
            department = st.selectbox("Department", _DEPT_CATS)
        with role_col:
            role = st.selectbox("Role", sorted(market["Role"].unique()))

        loc_col, type_col, edu_col = st.columns(3)
        with loc_col:
            work_location = st.selectbox("Work Location", _LOC_CATS)
        with type_col:
            emp_type = st.selectbox("Employment Type", _TYPE_CATS)
        with edu_col:
            education = st.selectbox(
                "Education Level",
                ["High School", "Associate", "Bachelor", "Master", "PhD"],
                index=2,
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Compensation
        comp_c1, comp_c2, comp_c3 = st.columns(3)
        with comp_c1:
            tenure = st.number_input("Tenure (years)", min_value=0.0, max_value=40.0,
                                     value=3.0, step=0.5)
        with comp_c2:
            base_salary = st.number_input("Base Salary ($)", min_value=30_000,
                                          max_value=500_000, value=85_000, step=1_000)
        with comp_c3:
            bonus = st.number_input("Bonus ($)", min_value=0, max_value=200_000,
                                    value=8_000, step=500)

        comp_c4, comp_c5, _ = st.columns(3)
        with comp_c4:
            stock = st.number_input("Stock Options ($)", min_value=0, max_value=500_000,
                                    value=0, step=500)
        with comp_c5:
            total_comp = st.number_input(
                "Total Compensation ($)", min_value=30_000, max_value=700_000,
                value=base_salary + bonus + stock, step=1_000,
                help="Defaults to Base + Bonus + Stock. Override if needed.",
            )

        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

        # Survey scores
        sh("Survey Scores  (1 = lowest, 5 = highest)")
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            job_sat = st.slider("Job Satisfaction", 1.0, 5.0, 3.8, 0.1)
        with s2:
            wlb = st.slider("Work-Life Balance", 1.0, 5.0, 3.8, 0.1)
        with s3:
            mgmt = st.slider("Management Support", 1.0, 5.0, 3.8, 0.1)
        with s4:
            career = st.slider("Career Development", 1.0, 5.0, 3.8, 0.1)
        with s5:
            engagement = st.slider("Engagement Level", 1.0, 5.0, 3.6, 0.1)

        submitted = st.form_submit_button("Score Employee", use_container_width=True)

    # ── Run inference on submit ──────────────────────────────────────────────
    if submitted:
        # Auto-calculate Compa_Ratio from market table
        benchmark = market_lookup.get((role, work_location), None)
        compa_ratio = (base_salary / benchmark) if benchmark else 1.0

        row_data = {
            "Tenure_Years":        tenure,
            "Education_Level":     _EDU_MAP.get(education, 3),
            "Job_Satisfaction":    job_sat,
            "Work_Life_Balance":   wlb,
            "Management_Support":  mgmt,
            "Career_Development":  career,
            "Engagement_Level":    engagement,
            "Base_Salary":         base_salary,
            "Bonus":               bonus,
            "Stock_Options":       stock,
            "Total_Compensation":  total_comp,
            "Compa_Ratio":         compa_ratio,
            "Feedback_Sentiment":  0.0,
            "Department":          department,
            "Employment_Type":     emp_type,
            "Work_Location":       work_location,
        }
        # Build OHE columns
        for d in _DEPT_CATS:
            row_data[f"Department_{d}"] = int(department == d)
        for t in _TYPE_CATS:
            row_data[f"Employment_Type_{t}"] = int(emp_type == t)
        for l in _LOC_CATS:
            row_data[f"Work_Location_{l}"] = int(work_location == l)

        X_row = pd.DataFrame([row_data])[MODEL_FEATURES]

        try:
            model = joblib.load(OUT / "catboost_best_model.joblib")
            prob  = float(model.predict_proba(X_row)[0, 1])
            tier  = _assign_tier(prob, threshold)
            est_cost = base_salary * 1.5 * prob

            st.session_state.scored_employees.append({
                "Employee ID":        emp_id,
                "Department":         department,
                "Role":               role,
                "Location":           work_location,
                "Tenure (yr)":        tenure,
                "Base Salary":        base_salary,
                "Compa-Ratio":        round(compa_ratio, 2),
                "Attrition Prob":     round(prob, 4),
                "Risk Tier":          tier,
                "Est. Cost":          round(est_cost, 0),
            })
            tier_color = RISK_COLORS.get(tier, GREY)
            st.markdown(
                f"<div style='background:{tier_color}18;border-left:4px solid {tier_color};"
                f"border-radius:6px;padding:0.8rem 1.2rem;margin-top:0.5rem'>"
                f"<b style='color:{tier_color}'>{tier} Risk</b>"
                f" &nbsp;|&nbsp; Attrition probability: "
                f"<b style='color:{BLACK}'>{prob:.1%}</b>"
                f" &nbsp;|&nbsp; Est. replacement cost: "
                f"<b style='color:{BLACK}'>${est_cost:,.0f}</b>"
                f"{'  |  Compa-ratio: <b>' + str(round(compa_ratio,2)) + '</b>' if benchmark else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Inference failed: {exc}")

    # ── Accumulated results table ────────────────────────────────────────────
    if st.session_state.scored_employees:
        st.markdown("<br>", unsafe_allow_html=True)
        sh(f"Scored Employees  ({len(st.session_state.scored_employees)} total)")

        results_df = pd.DataFrame(st.session_state.scored_employees)
        st.dataframe(
            results_df,
            hide_index=True,
            use_container_width=True,
            height=min(400, 55 + len(results_df) * 36),
            column_config={
                "Attrition Prob": st.column_config.ProgressColumn(
                    "Attrition Prob", format="%.1%", min_value=0, max_value=1,
                ),
                "Est. Cost": st.column_config.NumberColumn(
                    "Est. Cost", format="$%,.0f",
                ),
                "Base Salary": st.column_config.NumberColumn(
                    "Base Salary", format="$%,.0f",
                ),
            },
        )

        dl_col, clear_col, _ = st.columns([1, 1, 4])
        with dl_col:
            st.download_button(
                label="Download CSV",
                data=results_df.to_csv(index=False).encode(),
                file_name="scored_employees.csv",
                mime="text/csv",
            )
        with clear_col:
            if st.button("Clear Results"):
                st.session_state.scored_employees = []
                st.rerun()



def page_model_performance(eval_df: pd.DataFrame, winner: dict) -> None:
    """
    Model Performance: three-model scorecard, metric cards for the champion,
    radar comparison, lift explanation, and a threshold decision guide.
    """
    winner_model = winner["winner_model"]
    threshold    = winner["threshold"]

    # ── Champion metric cards ─────────────────────────────────────────────────
    sh(f"Champion Model — {winner_model}")
    champ = eval_df[eval_df["model"] == winner_model].iloc[0]

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        kpi("ROC-AUC",   f"{champ['test_roc_auc']:.4f}", "test set")
    with m2:
        kpi("PR-AUC",    f"{champ['test_pr_auc']:.4f}",  "test set")
    with m3:
        kpi("F1 Score",  f"{champ['test_f1']:.4f}",      "at threshold")
    with m4:
        kpi("Brier Score", f"{champ['test_brier']:.4f}", "lower is better")
    with m5:
        kpi("Lift @ Top 20%", f"{champ['lift_at_20pct']:.2f}x",
            "vs random baseline")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scorecard table ───────────────────────────────────────────────────────
    sh("Full Model Scorecard")

    # Highlight champion row via a styled html table
    col_labels = {
        "model":         "Model",
        "threshold":     "Threshold",
        "val_best_f1":   "Val F1",
        "test_roc_auc":  "ROC-AUC",
        "test_pr_auc":   "PR-AUC",
        "test_f1":       "Test F1",
        "test_brier":    "Brier",
        "lift_at_20pct": "Lift @20%",
    }
    fmt = {
        "threshold":     "{:.4f}",
        "val_best_f1":   "{:.4f}",
        "test_roc_auc":  "{:.4f}",
        "test_pr_auc":   "{:.4f}",
        "test_f1":       "{:.4f}",
        "test_brier":    "{:.4f}",
        "lift_at_20pct": "{:.3f}",
    }
    header_cells = "".join(
        f"<th style='padding:8px 14px;text-align:left;font-size:0.78rem;"
        f"text-transform:uppercase;letter-spacing:0.06em;color:{GREY};"
        f"border-bottom:2px solid {ORANGE}'>{v}</th>"
        for v in col_labels.values()
    )
    rows_html = ""
    for _, row in eval_df.iterrows():
        is_winner = row["model"] == winner_model
        bg = f"background:{GREY_LIGHT};" if is_winner else ""
        badge = (
            f" <span style='background:{ORANGE};color:{WHITE};border-radius:3px;"
            f"padding:1px 6px;font-size:0.68rem;font-weight:700;margin-left:6px'>"
            f"WINNER</span>"
        ) if is_winner else ""
        cells = "".join(
            f"<td style='padding:8px 14px;font-size:0.82rem;color:{BLACK};"
            f"border-bottom:1px solid #E5E7EB'>{row[c]}{badge if c=='model' else ''}</td>"
            if c == "model" else
            f"<td style='padding:8px 14px;font-size:0.82rem;color:{BLACK};"
            f"border-bottom:1px solid #E5E7EB'>{fmt[c].format(row[c])}</td>"
            for c in col_labels
        )
        rows_html += f"<tr style='{bg}'>{cells}</tr>"

    st.markdown(
        f"<div style='overflow-x:auto'><table style='border-collapse:collapse;"
        f"width:100%;background:{WHITE};border-radius:8px;"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.07)'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Side-by-side charts ───────────────────────────────────────────────────
    col_radar, col_bar = st.columns([1, 1])

    with col_radar:
        sh("Metric Radar — All Models")
        # Normalise each metric to 0-1 range for radar comparison
        radar_metrics = ["test_roc_auc", "test_pr_auc", "test_f1", "lift_at_20pct"]
        radar_labels  = ["ROC-AUC", "PR-AUC", "F1", "Lift @20%"]
        # Brier: lower is better, invert
        brier_inv = 1 - eval_df["test_brier"]
        radar_metrics_full  = radar_metrics + ["inv_brier"]
        radar_labels_full   = radar_labels  + ["1 - Brier"]
        eval_r = eval_df.copy()
        eval_r["inv_brier"] = brier_inv

        # Min-max scale each metric across models
        for col in radar_metrics_full:
            mn, mx = eval_r[col].min(), eval_r[col].max()
            eval_r[f"{col}_norm"] = (eval_r[col] - mn) / (mx - mn + 1e-9)

        model_colors = {"CatBoost": ORANGE, "RandomForest": GREY, "XGBoost": RED}
        fig = go.Figure()
        for _, row in eval_r.iterrows():
            vals = [row[f"{c}_norm"] for c in radar_metrics_full]
            vals += [vals[0]]  # close polygon
            lbl  = radar_labels_full + [radar_labels_full[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=lbl, fill="toself",
                name=row["model"],
                line=dict(color=model_colors.get(row["model"], GREY), width=2),
                fillcolor=model_colors.get(row["model"], GREY).replace("#", "rgba(") + ",0.1)"
                if False else "rgba(0,0,0,0)",
                opacity=0.85,
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#E5E7EB"),
                angularaxis=dict(gridcolor="#E5E7EB"),
                bgcolor=WHITE,
            ),
            legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            margin=dict(t=20, b=40, l=20, r=20), height=320,
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_bar:
        sh("ROC-AUC & PR-AUC by Model")
        fig = go.Figure()
        bar_colors = [model_colors.get(m, GREY) for m in eval_df["model"]]
        fig.add_trace(go.Bar(
            x=eval_df["model"], y=eval_df["test_roc_auc"],
            name="ROC-AUC", marker_color=bar_colors, opacity=1.0,
            text=[f"{v:.4f}" for v in eval_df["test_roc_auc"]],
            textposition="outside",
            yaxis="y",
            hovertemplate="%{x}  ROC-AUC: %{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=eval_df["model"], y=eval_df["test_pr_auc"],
            name="PR-AUC", marker_color=bar_colors, opacity=0.55,
            text=[f"{v:.4f}" for v in eval_df["test_pr_auc"]],
            textposition="outside",
            yaxis="y",
            hovertemplate="%{x}  PR-AUC: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            barmode="group",
            xaxis=dict(showgrid=False),
            yaxis=dict(range=[0.55, 0.92], gridcolor=GREY_LIGHT, title="Score"),
            legend=dict(orientation="h", y=1.08),
            margin=dict(t=30, b=10, l=10, r=10), height=320,
            **_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Threshold guide ───────────────────────────────────────────────────────
    sh("Decision Threshold Guide")
    st.markdown(
        f"<div style='font-size:0.83rem;color:{GREY};max-width:720px;margin-bottom:1rem'>"
        f"The deployed threshold of <b style='color:{BLACK}'>{threshold:.4f}</b> was chosen by "
        f"maximising F1 on the validation set (val F1 = {winner['val_best_f1']:.4f}). "
        f"A lower threshold captures more at-risk employees (higher recall, more false positives). "
        f"A higher threshold is more precise but misses borderline cases."
        "</div>",
        unsafe_allow_html=True,
    )

    # Precision / Recall trade-off illustration using the scored probability distribution
    t_vals = [i / 100 for i in range(20, 80)]
    # Load pre-scored probabilities for illustration
    risk_df  = pd.read_csv(OUT / "risk_scored_employees.csv")

    # Compute approximate precision + recall by comparing to Attrition_Target
    # We need actual labels — load from attrition_data
    att = pd.read_csv(DATA / "attrition_data.csv")
    att["Attrition_Target"] = (att["Attrition_Status"] == "Yes").astype(int)
    # align by positional index (same merge order as load_master)
    labels = att["Attrition_Target"].values
    probs_arr = risk_df.sort_values("employee_idx")["attrition_probability"].values

    precisions, recalls, f1s = [], [], []
    for t in t_vals:
        preds = (probs_arr >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_vals, y=precisions, mode="lines", name="Precision",
        line=dict(color=ORANGE, width=2),
        hovertemplate="Threshold %{x:.2f}  Precision %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t_vals, y=recalls, mode="lines", name="Recall",
        line=dict(color=RED, width=2),
        hovertemplate="Threshold %{x:.2f}  Recall %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t_vals, y=f1s, mode="lines", name="F1",
        line=dict(color=BLACK, width=2, dash="dot"),
        hovertemplate="Threshold %{x:.2f}  F1 %{y:.2%}<extra></extra>",
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color=GREY, line_width=1.5,
        annotation_text=f"Deployed {threshold:.4f}",
        annotation_font_color=GREY,
    )
    fig.update_layout(
        xaxis=dict(title="Decision Threshold", gridcolor=GREY_LIGHT),
        yaxis=dict(title="Score", tickformat=".0%", gridcolor=GREY_LIGHT),
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=20, b=10, l=10, r=10), height=300,
        **_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _placeholder(page_name: str) -> None:
    """Stub rendered for pages that will be implemented in later phases."""
    st.info(f"**{page_name}** will be available in the next implementation phase.")


# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

# Sidebar label → internal page key
PAGE_MAP: dict[str, str] = {
    "01  Overview":           "Overview",
    "02  Risk Table":         "Risk Table",
    "03  Analytics Explorer": "Analytics",
    "04  Retention Planner":  "Retention Planner",
    "05  Score New Data":     "Score New Data",
    "06  Model Performance":  "Model Performance",
}


def main() -> None:
    inject_css()

    df      = load_master()
    winner  = load_winner_meta()
    matrix  = load_retention_matrix()
    eval_df = load_model_eval()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## HR Attrition")
        st.caption(
            f"{len(df):,} employees scored  |  Model: {winner['winner_model']}"
        )
        st.markdown("---")
        selected = st.radio(
            "Section",
            list(PAGE_MAP.keys()),
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption("Prototype — not for production use.")

    page_key = PAGE_MAP[selected]

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(f"## {selected}")
    st.markdown("---")

    # ── Routing ───────────────────────────────────────────────────────────────
    if page_key == "Overview":
        page_overview(df)
    elif page_key == "Risk Table":
        page_risk_table(df)
    elif page_key == "Analytics":
        page_analytics(df)
    elif page_key == "Retention Planner":
        page_retention_planner(df, matrix)
    elif page_key == "Score New Data":
        page_score_new_data(winner)
    elif page_key == "Model Performance":
        page_model_performance(eval_df, winner)


if __name__ == "__main__":
    main()
