import html
import json
import os
from datetime import datetime, timezone
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import norm

from inverse_design import InverseDesignSystem

st.set_page_config(
    page_title="UHPC Inverse Design",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

SCM_OPTIONS = [
    "Silica Fume",
    "Fly Ash",
    "Limestone Powder",
    "Quartz Powder",
    "Glass Powder",
    "Rice Husk Ash",
    "Metakaolin",
    "GGBFS",
    "Steel Slag",
]

RISK_MODE_CONFIG = {
    "Aggressive (Beta=0.0)": {"beta": 0.0, "label": "Exploratory"},
    "Balanced (Beta=0.5)": {"beta": 0.5, "label": "Balanced"},
    "Conservative (Beta=2.0)": {"beta": 2.0, "label": "Conservative"},
}

# Academic & Minimalistic Palette (Navy, Slate, Steel, Teal)
CHART_PALETTE = [
    "#0F172A", # Deep Navy
    "#1D4ED8", # Academic Blue
    "#0369A1", # Steel Blue
    "#0F766E", # Muted Teal
    "#475569", # Slate Gray
    "#94A3B8", # Light Slate
    "#CBD5E1", # Silver
]

PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "academic_high_res",
        "height": 1080,
        "width": 1920,
        "scale": 4
    }
}


def safe_text(value):
    return html.escape(str(value))


def format_number(value, decimals=2):
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:,.{decimals}f}"


def format_timestamp(value):
    if not value:
        return "No run yet"
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone().strftime("%d %b %Y | %H:%M")


def to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def render_gap(height_rem=1.25):
    st.markdown(
        f'<div style="height:{height_rem}rem;"></div>',
        unsafe_allow_html=True,
    )

# --- MINIMALISTIC & ACADEMIC CSS ---
st.markdown(
    dedent(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

            :root {
                --app-bg: #FAFAFA;
                --panel-bg: #FFFFFF;
                --panel-border: #E2E8F0;
                --text-primary: #0F172A;
                --text-secondary: #475569;
                --accent-blue: #1D4ED8;
            }

            html, body, [class*="css"] {
                font-family: "Inter", -apple-system, sans-serif;
                background-color: var(--app-bg);
                color: var(--text-primary);
            }

            .stApp {
                background: var(--app-bg);
            }

            .block-container {
                padding-top: 2rem !important;
                padding-bottom: 2rem !important;
            }

            /* Form Elements - Square & Clean */
            [data-baseweb="input"] > div,
            [data-baseweb="select"] > div,
            .stNumberInput input,
            .stTextInput input {
                border-radius: 4px !important;
                border-color: #CBD5E1 !important;
                background: #FFFFFF !important;
            }

            .stButton > button {
                min-height: 2.8rem;
                border-radius: 6px;
                border: none;
                background: var(--text-primary);
                color: #FFFFFF;
                font-weight: 500;
                transition: opacity 0.2s;
            }

            .stButton > button:hover {
                background: var(--text-primary);
                opacity: 0.85;
            }

            /* Dashboard Headers */
            .header-card {
                background: #FFFFFF;
                border: 1px solid var(--panel-border);
                border-radius: 8px;
                padding: 1.5rem 2rem;
            }

            .summary-card {
                background: #FFFFFF;
                border: 1px solid var(--panel-border);
                border-radius: 8px;
                padding: 1.25rem 1.5rem;
                height: 100%;
            }

            .dashboard-kicker {
                color: var(--text-secondary);
                letter-spacing: 0.05em;
                font-size: 0.8rem;
                font-weight: 600;
                text-transform: uppercase;
                margin-bottom: 0.4rem;
            }

            .dashboard-title {
                color: var(--text-primary);
                font-size: 2.2rem;
                font-weight: 700;
                margin: 0;
            }

            .dashboard-subtitle {
                color: var(--text-secondary);
                font-size: 1rem;
                margin: 0.5rem 0 0;
            }

            .badge-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 1rem;
            }

            .st-badge {
                background-color: #F1F5F9;
                color: var(--text-secondary);
                padding: 0.2rem 0.6rem;
                border-radius: 4px;
                font-size: 0.8rem;
                font-weight: 500;
                border: 1px solid #E2E8F0;
            }

            /* Summary Grid */
            .summary-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 1px;
                background: var(--panel-border);
                border: 1px solid var(--panel-border);
                border-radius: 6px;
                overflow: hidden;
                margin-top: 1rem;
            }

            .summary-item {
                background: #FFFFFF;
                padding: 0.75rem 1rem;
            }

            .summary-label {
                display: block;
                color: var(--text-secondary);
                font-size: 0.7rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }

            .summary-value {
                display: block;
                color: var(--text-primary);
                font-size: 0.95rem;
                font-weight: 600;
                margin-top: 0.2rem;
            }

            /* Metrics Styling */
            [data-testid="stMetric"] {
                background: #FFFFFF;
                border: 1px solid #E2E8F0 !important;
                border-left: 4px solid var(--accent-blue) !important;
                border-radius: 6px;
                padding: 1rem 1.25rem;
                min-height: 7rem;
            }

            [data-testid="stMetricLabel"] {
                color: var(--text-secondary);
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.75rem;
            }

            /* Chart Containers - FIX FOR OVERLAPPING */
            div[data-testid="stPlotlyChart"],
            div[data-testid="stDataFrame"] {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 6px;
                padding: 0.5rem;
                box-sizing: border-box !important; /* Critical fix */
                width: 100%;
                margin-bottom: 1rem;
            }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading inverse design system...")
def load_system():
    try:
        return InverseDesignSystem()
    except Exception as exc:
        st.error(f"Error loading system: {exc}")
        return None


system = load_system()


def get_options(category):
    if system and "Categorical_Mappings" in system.materials_data:
        mapping = system.materials_data["Categorical_Mappings"].get(category, {})
        return sorted(list(mapping.keys()))
    return []


def select_or_none(label, options):
    available = options or ["Not specified"]
    selected = st.selectbox(label, available, index=0)
    return None if selected == "Not specified" else selected


def build_mix_dataframe(mix):
    if not mix:
        return pd.DataFrame(columns=["Component", "Amount (kg/m3)", "Share (%)"])

    df = pd.DataFrame(list(mix.items()), columns=["Component", "Amount (kg/m3)"])
    df["Amount (kg/m3)"] = pd.to_numeric(df["Amount (kg/m3)"], errors="coerce")
    df = df.dropna(subset=["Amount (kg/m3)"])
    df = df[df["Amount (kg/m3)"] > 0].sort_values("Amount (kg/m3)", ascending=False)
    df = df.reset_index(drop=True)

    total = df["Amount (kg/m3)"].sum()
    df["Share (%)"] = (df["Amount (kg/m3)"] / total * 100.0) if total > 0 else 0.0
    return df


def build_header_html(target_strength, risk_label, specimen):
    badges = [
        f"Target {format_number(target_strength, 1)} MPa",
        risk_label,
        specimen or "Specimen not set",
    ]
    badge_html = "".join(
        f'<span class="st-badge">{safe_text(item)}</span>' for item in badges
    )
    return dedent(
        f"""
        <div class="header-card">
            <div class="dashboard-kicker">Optimization Engine</div>
            <h1 class="dashboard-title">Inverse Mix Design</h1>
            <p class="dashboard-subtitle">Constraint-aware optimization and uncertainty analysis for ultra-high performance concrete.</p>
            <div class="badge-row">{badge_html}</div>
        </div>
        """
    )


def build_summary_card_html(latest_result, status_text, fallback_items):
    if latest_result:
        diagnostics = latest_result["diagnostics"]
        summary_title = format_timestamp(latest_result["payload"]["timestamp"])
        items = [
            ("Predicted", f"{format_number(latest_result['pred_strength'], 2)} MPa"),
            ("Uncertainty", f"± {format_number(latest_result['pred_uncertainty'], 2)} MPa"),
            ("Confidence", f"{format_number(diagnostics.get('confidence_score', 0.0), 1)}%"),
            ("Reliability", diagnostics.get("reliability_status", "Unknown")),
        ]
    else:
        summary_title = "Awaiting first run"
        items = fallback_items

    items_html = "".join(
        dedent(
            f"""
            <div class="summary-item">
                <span class="summary-label">{safe_text(label)}</span>
                <span class="summary-value">{safe_text(value)}</span>
            </div>
            """
        )
        for label, value in items
    )

    return dedent(
        f"""
        <div class="summary-card">
            <div class="dashboard-kicker">Run Summary &bull; {safe_text(summary_title)}</div>
            <div class="summary-grid">{items_html}</div>
        </div>
        """
    )


def render_section_header(title, copy=None):
    copy_html = f'<p style="color:#64748B; font-size:0.9rem; margin:0 0 1rem 0;">{safe_text(copy)}</p>' if copy else ""
    st.markdown(
        dedent(
            f"""
            <h3 style="color:#0F172A; font-size:1.15rem; font-weight:600; margin:0 0 0.2rem 0;">{safe_text(title)}</h3>
            {copy_html}
            """
        ),
        unsafe_allow_html=True,
    )


def build_distribution_figure(pred_strength, pred_uncertainty, target_strength):
    sigma = max(float(pred_uncertainty), 1e-6)
    mu = float(pred_strength)
    target = float(target_strength)
    x = np.linspace(max(0.0, mu - 4.5 * sigma), mu + 4.5 * sigma, 360)
    density = norm.pdf(x, loc=mu, scale=sigma)
    cumulative = norm.cdf(x, loc=mu, scale=sigma)
    interval_68 = (mu - sigma, mu + sigma)
    interval_95 = norm.interval(0.95, loc=mu, scale=sigma)
    exceedance_prob = max(0.0, (1.0 - norm.cdf(target, loc=mu, scale=sigma)) * 100.0)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Uncertainty bands
    fig.add_vrect(x0=interval_95[0], x1=interval_95[1], fillcolor="rgba(148, 163, 184, 0.15)", line_width=0, layer="below")
    fig.add_vrect(x0=interval_68[0], x1=interval_68[1], fillcolor="rgba(148, 163, 184, 0.25)", line_width=0, layer="below")

    fig.add_trace(
        go.Scatter(
            x=x, y=density, mode="lines", name="Posterior density",
            line=dict(color="#0F172A", width=2.5), fill="tozeroy", fillcolor="rgba(15, 23, 42, 0.05)",
            hovertemplate="Strength %{x:.2f} MPa<br>Density %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=cumulative, mode="lines", name="Cumulative probability",
            line=dict(color="#64748B", width=2, dash="dot"),
            hovertemplate="Strength %{x:.2f} MPa<br>CDF %{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_vline(x=target, line_color="#1D4ED8", line_width=2, line_dash="dash", annotation_text="Target", annotation_position="top right")
    fig.add_vline(x=mu, line_color="#0F172A", line_width=2, annotation_text="Mean", annotation_position="top left")

    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=1.15, showarrow=False, text=f"68% interval: {interval_68[0]:.2f} to {interval_68[1]:.2f} MPa", font=dict(size=12, color="#475569"), align="left")
    fig.add_annotation(xref="paper", yref="paper", x=0.99, y=1.15, showarrow=False, text=f"P(f'c >= target): {exceedance_prob:.1f}%", font=dict(size=12, color="#475569"), align="right", xanchor="right")

    fig.update_xaxes(title_text="Compressive Strength (MPa)", showgrid=True, gridcolor="#E2E8F0", zeroline=False)
    fig.update_yaxes(title_text="Probability Density", showgrid=True, gridcolor="#E2E8F0", zeroline=False, secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Probability", range=[0, 1.05], showgrid=False, zeroline=False, secondary_y=True)
    
    # Explicit height fixes overlap bug
    fig.update_layout(
        height=450, 
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        font=dict(family="Inter, sans-serif", color="#0F172A"),
        hovermode="x unified",
    )
    return fig


def build_composition_figure(mix_df):
    plot_df = mix_df.copy()
    if len(plot_df) > 6:
        retained = plot_df.head(5).copy()
        other_row = pd.DataFrame([{"Component": "Other", "Amount (kg/m3)": plot_df.iloc[5:]["Amount (kg/m3)"].sum(), "Share (%)": plot_df.iloc[5:]["Share (%)"].sum()}])
        plot_df = pd.concat([retained, other_row], ignore_index=True)

    fig = go.Figure(
        go.Pie(
            labels=plot_df["Component"], values=plot_df["Amount (kg/m3)"],
            hole=0.65, sort=False,
            marker=dict(colors=CHART_PALETTE[: len(plot_df)], line=dict(color='#FFFFFF', width=2)),
            textinfo="percent", textfont=dict(size=12, color="#FFFFFF"),
            hovertemplate="%{label}<br>%{value:.2f} kg/m3<br>%{percent}<extra></extra>",
        )
    )
    # Explicit height fixes overlap bug
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="v", x=1.0, y=0.5, font=dict(color="#475569")),
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    return fig


def build_component_bar_figure(mix_df):
    plot_df = mix_df.head(8).iloc[::-1]

    fig = go.Figure(
        go.Bar(
            x=plot_df["Amount (kg/m3)"], y=plot_df["Component"], orientation="h",
            marker=dict(
                color=plot_df["Amount (kg/m3)"],
                colorscale=[[0.0, "#94A3B8"], [0.5, "#1D4ED8"], [1.0, "#0F172A"]], # Academic colorscale
                showscale=False,
            ),
            hovertemplate="%{y}<br>%{x:.2f} kg/m3<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Amount (kg/m3)", showgrid=True, gridcolor="#E2E8F0")
    fig.update_yaxes(title_text="", showgrid=False)
    
    # Explicit height fixes overlap bug
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0F172A"),
    )
    return fig


filler_options = get_options("Type of Filler")
fiber_options = get_options("Type of Fiber")
curing_options = get_options("Curing")
spec_options = get_options("Specimen Size")

if "mix_results" not in st.session_state:
    st.session_state.mix_results = None
if "system_status" not in st.session_state:
    st.session_state.system_status = "Ready" if system else "Unavailable"


with st.sidebar:
    st.markdown('<h3 style="font-size:1.1rem; margin-bottom:0.5rem; color:#0F172A;">Design Inputs</h3>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.85rem; color:#64748B; margin-bottom:1.5rem;">Configure material constraints below.</p>', unsafe_allow_html=True)

    target_strength = st.number_input("Target Strength (MPa)", value=150.0, step=5.0)
    risk_mode = st.selectbox("Risk Strategy", list(RISK_MODE_CONFIG.keys()), index=1)
    specimen = select_or_none("Specimen Geometry", spec_options)
    binder = st.selectbox("Primary Binder", ["Cement"])
    scm_list = st.multiselect("SCMs", SCM_OPTIONS, default=["Silica Fume"])
    filler_type = select_or_none("Filler", filler_options)
    fiber_type = select_or_none("Fiber", fiber_options)
    curing = select_or_none("Curing Regime", curing_options)
    run_button = st.button("Run Optimization", type="primary", use_container_width=True)


if run_button:
    if not system:
        st.session_state.system_status = "Unavailable"
        st.error("The inverse design system is unavailable.")
    else:
        with st.spinner("Running optimization..."):
            try:
                beta = RISK_MODE_CONFIG[risk_mode]["beta"]
                mix, pred_strength, pred_uncertainty, diagnostics = system.optimize_mix(
                    target_strength=float(target_strength),
                    binder_type=binder,
                    scm_types=scm_list or [],
                    filler_type=filler_type,
                    fiber_type=fiber_type,
                    curing_type=curing,
                    specimen_size=specimen,
                    beta=beta,
                    return_diagnostics=True,
                )

                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "inputs": {
                        "target_strength": float(target_strength),
                        "risk_mode": risk_mode,
                        "binder": binder,
                        "specimen": specimen,
                        "scm_list": scm_list or [],
                        "filler_type": filler_type,
                        "fiber_type": fiber_type,
                        "curing": curing,
                    },
                    "mix": to_jsonable(mix),
                    "diagnostics": to_jsonable(diagnostics),
                }

                st.session_state.mix_results = {
                    "mix": mix,
                    "pred_strength": float(pred_strength),
                    "pred_uncertainty": float(pred_uncertainty),
                    "diagnostics": diagnostics,
                    "payload": payload,
                }
                st.session_state.system_status = "Updated"
            except Exception as exc:
                st.session_state.system_status = "Failed"
                st.error(f"Optimization error: {exc}")


latest_result = st.session_state.mix_results

display_inputs = (
    latest_result["payload"]["inputs"]
    if latest_result
    else {
        "target_strength": float(target_strength),
        "risk_mode": risk_mode,
        "binder": binder,
        "specimen": specimen,
        "scm_list": scm_list or [],
        "filler_type": filler_type,
        "fiber_type": fiber_type,
        "curing": curing,
    }
)

display_target = float(display_inputs["target_strength"])
display_risk_mode = display_inputs["risk_mode"]
display_binder = display_inputs["binder"]
display_specimen = display_inputs.get("specimen")
display_scm_list = display_inputs.get("scm_list") or []

header_col, summary_col = st.columns([1.5, 1], gap="large")

with header_col:
    st.markdown(
        build_header_html(display_target, RISK_MODE_CONFIG[display_risk_mode]["label"], display_specimen),
        unsafe_allow_html=True,
    )

with summary_col:
    st.markdown(
        build_summary_card_html(
            latest_result, st.session_state.system_status,
            [("Risk", RISK_MODE_CONFIG[display_risk_mode]["label"]), ("System", "Loaded" if system else "Unavailable")]
        ),
        unsafe_allow_html=True,
    )

render_gap(1.5)

metric_cols = st.columns(4, gap="medium")
if latest_result:
    diagnostics = latest_result["diagnostics"]
    with metric_cols[0]: st.metric("Target Strength", f"{format_number(display_target, 1)} MPa")
    with metric_cols[1]: st.metric("Predicted Strength", f"{format_number(latest_result['pred_strength'], 2)} MPa", f"{latest_result['pred_strength'] - display_target:+.2f} MPa", delta_color="off")
    with metric_cols[2]: st.metric("Prediction Uncertainty", f"± {format_number(latest_result['pred_uncertainty'], 2)} MPa")
    with metric_cols[3]: st.metric("Confidence Score", f"{format_number(diagnostics.get('confidence_score', 0.0), 1)}%", diagnostics.get("reliability_status", "Unknown"), delta_color="off")

render_gap(1.5)

if latest_result:
    mix_df = build_mix_dataframe(latest_result["mix"])

    if mix_df.empty:
        st.warning("The optimization finished, but no positive component quantities were returned.")
    else:
        # ROW 1: Distribution Plot (Full Width)
        with st.container():
            render_section_header("Strength Distribution", "Posterior density, cumulative probability, and target alignment.")
            st.plotly_chart(
                build_distribution_figure(latest_result["pred_strength"], latest_result["pred_uncertainty"], display_target),
                use_container_width=True, config=PLOTLY_CONFIG,
            )

        render_gap(1.5)
        
        # ROW 2: Composition and Ranking (Side by Side to prevent layout overlap)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            render_section_header("Composition Profile", "Grouped share of the optimized mix inventory.")
            st.plotly_chart(
                build_composition_figure(mix_df),
                use_container_width=True, config=PLOTLY_CONFIG,
            )

        with col2:
            render_section_header("Component Ranking", "Highest-volume constituents in the optimized mix.")
            st.plotly_chart(
                build_component_bar_figure(mix_df),
                use_container_width=True, config=PLOTLY_CONFIG,
            )

        render_gap(1.5)
        
        # ROW 3: Data Table
        with st.container():
            render_section_header("Mix Data Output", "Raw quantities and proportional shares.")
            st.dataframe(
                mix_df[["Component", "Amount (kg/m3)", "Share (%)"]].head(8),
                hide_index=True, use_container_width=True,
            )
else:
    st.info("Configure the design inputs in the sidebar and run the optimization to populate the dashboard.")