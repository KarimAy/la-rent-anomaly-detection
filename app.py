"""
Streamlit dashboard — LA Rent Price Anomaly Detection.
"""

import sys
import warnings
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.prediction_pipeline import (
    HISTORICAL_CUTOFF,
    detect_anomaly,
    get_features,
    get_kalman_bounds,
    load_models,
    predict_ensemble,
)

# ── Page config (must be the first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="LA Rent Anomaly Detector",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
NEIGHBORHOODS = [
    "Beverly Hills", "Boyle Heights", "Brentwood", "Culver City",
    "Downtown LA", "Exposition Park", "Glassell Park", "Hancock Park",
    "Highland Park", "Koreatown", "Leimert Park", "Los Feliz",
    "Mar Vista", "Mid-Wilshire", "Palms", "Santa Monica",
    "Silver Lake", "University Park", "Vermont Square", "West Adams",
    "West Hollywood", "Westwood",
]

PRICE_TIERS = {
    "Tier 1 · Lower-income": [
        "University Park", "Exposition Park", "Vermont Square",
    ],
    "Tier 2 · Mid-range": [
        "Koreatown", "West Adams", "Leimert Park", "Boyle Heights",
        "Silver Lake", "Los Feliz", "Highland Park", "Glassell Park",
        "Palms", "Mid-Wilshire", "Downtown LA",
    ],
    "Tier 3 · Mid-high": [
        "Culver City", "Mar Vista", "West Hollywood", "Hancock Park", "Westwood",
    ],
    "Tier 4 · Premium": ["Beverly Hills", "Brentwood", "Santa Monica"],
}

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.badge {
    display: inline-block;
    padding: 3px 11px;
    border-radius: 11px;
    font-size: 0.80em;
    font-weight: 700;
    letter-spacing: 0.05em;
    vertical-align: middle;
}
.badge-high   { background: #198754; color: #fff; }
.badge-medium { background: #fd7e14; color: #fff; }
.badge-low    { background: #dc3545; color: #fff; }
.badge-future { background: #6f42c1; color: #fff; }

.model-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 14px 16px;
    height: 100%;
}
.model-card.ensemble { border-color: #adb5bd; background: #fff; }
.model-name { font-size: 0.76em; color: #6c757d; text-transform: uppercase;
              letter-spacing: 0.07em; margin-bottom: 3px; }
.model-pred { font-size: 1.55em; font-weight: 700; color: #212529; line-height: 1.1; }
.model-diff { font-size: 0.83em; margin-top: 5px; }
.model-meta { font-size: 0.72em; color: #adb5bd; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models (first run only)…")
def _get_models() -> dict:
    return load_models()


# ── Gauge chart ────────────────────────────────────────────────────────────────
def _make_gauge(
    user_rent: float,
    lower: float,
    upper: float,
    ensemble_pred: float,
    is_anomaly: bool,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.6), facecolor="white")
    ax.set_facecolor("white")

    all_vals = [lower, upper, user_rent, ensemble_pred]
    span = max(all_vals) - min(all_vals)
    pad  = max(span * 0.38, 350)
    x_min, x_max = min(all_vals) - pad, max(all_vals) + pad

    # Out-of-bounds zones (subtle red)
    ax.axvspan(x_min, lower, alpha=0.07, color="#dc3545", zorder=1)
    ax.axvspan(upper, x_max, alpha=0.07, color="#dc3545", zorder=1)

    # Normal zone (green fill)
    ax.axvspan(lower, upper, alpha=0.13, color="#198754", zorder=1, label="Normal range (2σ)")

    # Bound markers
    for bnd in (lower, upper):
        ax.axvline(bnd, color="#198754", alpha=0.55, linestyle="--", linewidth=1.4, zorder=2)
    ax.text(lower, 0.97, f"${lower:,.0f}", ha="center", va="top",
            fontsize=8, color="#198754", fontweight="bold", zorder=4)
    ax.text(upper, 0.97, f"${upper:,.0f}", ha="center", va="top",
            fontsize=8, color="#198754", fontweight="bold", zorder=4)

    # Predicted rent line
    ax.axvline(ensemble_pred, color="#1f77b4", linestyle=":", linewidth=2.2, zorder=3,
               label=f"Predicted ${ensemble_pred:,.0f}")
    ax.text(ensemble_pred, 0.60, f"Predicted\n${ensemble_pred:,.0f}",
            ha="center", va="bottom", fontsize=7.8, color="#1f77b4",
            fontweight="bold", linespacing=1.35, zorder=4)

    # User rent marker
    u_color = "#dc3545" if is_anomaly else "#198754"
    ax.axvline(user_rent, color=u_color, linestyle="-", linewidth=2.6, zorder=3,
               label=f"Your Rent ${user_rent:,.0f}")
    ax.plot(user_rent, 0.35, "D", color=u_color, markersize=12, zorder=5)
    ax.text(user_rent, 0.05, f"Your Rent\n${user_rent:,.0f}",
            ha="center", va="bottom", fontsize=7.8, color=u_color,
            fontweight="bold", linespacing=1.35, zorder=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.tick_params(axis="x", labelsize=8.5, colors="#6c757d")

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#dee2e6")

    fig.tight_layout(pad=0.5)
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏠 LA Rent Detector")
    st.caption("Is a rent price normal or anomalous for its neighborhood and time period?")
    st.divider()

    neighborhood = st.selectbox(
        "Neighborhood",
        NEIGHBORHOODS,
        index=NEIGHBORHOODS.index("Koreatown"),
    )

    selected_date = st.date_input(
        "Month",
        value=date(2024, 6, 30),
        min_value=date(2020, 1, 1),
        help=(
            "Select any day within the target month. "
            "Historical: 2020–2025 (features from master_model.csv). "
            "Future: live BLS + LA City API data."
        ),
    )

    user_rent = st.number_input(
        "Monthly Rent ($)",
        min_value=500,
        max_value=25_000,
        value=2_600,
        step=50,
        help="The rent price you want to evaluate.",
    )

    st.divider()
    analyze_btn = st.button("🔍 Analyze Rent", type="primary", use_container_width=True)
    st.divider()

    with st.expander("ℹ️ About", expanded=False):
        st.markdown("""
**Models**
- XGBoost (MAPE 4.92%) — tabular economic features, neighborhood dummies
- Prophet (MAPE 4.45%) — per-neighborhood trend + seasonality
- LSTM (MAPE 3.97%) — 12-month sequence with neighborhood embeddings
- Ensemble (MAPE 2.66%) — inverse-MAPE weighted average
- Kalman Filter — dynamic 2σ anomaly bounds on ensemble residuals

**Data sources**
- Rent: Zillow ZORI (2020–2025)
- Permits: LA City Open Data
- Employment: BLS (LA–Long Beach–Anaheim MSA)
- Demographics: Census ACS 5-year estimates

**Confidence levels**
- **HIGH** — historical date, exact row in training data
- **MEDIUM** — future date, all live APIs succeeded
- **LOW** — future date, API fallbacks used
        """)

    with st.expander("📍 Neighborhoods by Tier", expanded=False):
        for tier, hoods in PRICE_TIERS.items():
            st.markdown(f"**{tier}**")
            st.markdown("  " + " · ".join(hoods))


# ── Load models ────────────────────────────────────────────────────────────────
try:
    models = _get_models()
except Exception as _load_err:
    st.error(f"**Model loading failed:** {_load_err}")
    st.info(
        "Ensure all files exist in `models/` and required packages are installed "
        "(xgboost, prophet, torch, scikit-learn, joblib)."
    )
    st.stop()


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("LA Rent Price Anomaly Detection")
st.caption(
    "Learns what rent *should* be from economic fundamentals (permits, employment, demographics), "
    "then flags months where actual rent deviates unexpectedly — a signal worth investigating."
)


# ── Run analysis on button click ───────────────────────────────────────────────
if analyze_btn:
    # Clear previous state
    st.session_state.pop("result", None)
    st.session_state.pop("error", None)

    with st.spinner("Running ensemble prediction…"):
        try:
            ts_date   = pd.Timestamp(selected_date)
            is_future = ts_date > HISTORICAL_CUTOFF

            # 1. Get features
            features, confidence = get_features(neighborhood, ts_date)

            # 2. Ensemble prediction
            preds         = predict_ensemble(models, features, neighborhood, ts_date)
            ensemble_pred = preds["predicted_rent"]

            # 3. Kalman bounds (with graceful fallback for pre-2022 dates)
            bounds_note = None
            try:
                bounds = get_kalman_bounds(models, neighborhood, ts_date, ensemble_pred)
            except ValueError:
                # anomaly_results.csv starts 2022-01; for earlier dates use last known state
                kdf = models["kalman_last_state"]
                krow = kdf[kdf["Neighborhood"] == neighborhood]
                std = float(krow["smoothed_std"].iloc[0])
                bounds = {
                    "lower": ensemble_pred - 2.0 * std,
                    "upper": ensemble_pred + 2.0 * std,
                }
                bounds_note = (
                    "Exact Kalman bounds are not available for this period "
                    "(anomaly_results.csv begins 2022-01). "
                    "Showing estimated ±2σ bounds using the last fitted Kalman state."
                )

            # 4. Anomaly detection
            anomaly_result = detect_anomaly(
                float(user_rent), bounds["lower"], bounds["upper"], ensemble_pred
            )

            st.session_state["result"] = {
                "neighborhood": neighborhood,
                "date":         selected_date,
                "user_rent":    float(user_rent),
                "confidence":   confidence,
                "preds":        preds,
                "bounds":       bounds,
                "bounds_note":  bounds_note,
                "anomaly":      anomaly_result,
                "is_future":    is_future,
                "features":     features,
            }

        except Exception as exc:
            st.session_state["error"] = str(exc)


# ── Render: error ──────────────────────────────────────────────────────────────
if st.session_state.get("error"):
    st.error(f"**Analysis failed:** {st.session_state['error']}")
    st.caption(
        "Common causes: neighborhood with limited historical data selected for an early date "
        "(e.g. Exposition Park before 2022), or a live API timeout for future dates."
    )


# ── Render: result ─────────────────────────────────────────────────────────────
elif "result" in st.session_state:
    r          = st.session_state["result"]
    anomaly    = r["anomaly"]
    preds      = r["preds"]
    bounds     = r["bounds"]
    is_anomaly = anomaly["is_anomaly"]
    conf       = r["confidence"]
    mode_label = "🔮 Future" if r["is_future"] else "📚 Historical"

    # ── Section header ──
    left_hdr, right_hdr = st.columns([5, 2])
    with left_hdr:
        st.markdown(
            f"### {r['neighborhood']} &nbsp;·&nbsp; "
            f"{pd.Timestamp(r['date']).strftime('%B %Y')}"
        )
    with right_hdr:
        st.markdown(
            f"<div style='text-align:right; padding-top:6px;'>"
            f"<span class='badge badge-{'future' if r['is_future'] else 'high' if conf=='high' else conf}'>"
            f"{mode_label}</span> &nbsp; "
            f"<span class='badge badge-{conf}'>{conf.upper()} CONFIDENCE</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if r.get("bounds_note"):
        st.caption(f"ℹ️ {r['bounds_note']}")

    st.divider()

    # ── Anomaly / normal banner ──
    if is_anomaly:
        direction = anomaly["direction"]
        delta_pct = (
            abs(r["user_rent"] - preds["predicted_rent"])
            / preds["predicted_rent"] * 100
        )
        st.error(
            f"🚨 **ANOMALY DETECTED** — "
            f"Your rent of **${r['user_rent']:,.0f}** is "
            f"**{anomaly['anomaly_score']:.2f}σ {direction}** the model prediction "
            f"of **${preds['predicted_rent']:,.0f}** ({delta_pct:.1f}% deviation). "
            f"This falls outside the Kalman 2σ confidence interval "
            f"[**${bounds['lower']:,.0f} – ${bounds['upper']:,.0f}**]."
        )
    else:
        st.success(
            f"✅ **NORMAL** — "
            f"Your rent of **${r['user_rent']:,.0f}** is within the expected range "
            f"[**${bounds['lower']:,.0f} – ${bounds['upper']:,.0f}**] "
            f"for {r['neighborhood']} in {pd.Timestamp(r['date']).strftime('%B %Y')}. "
            f"Anomaly score: **{anomaly['anomaly_score']:.2f}σ** "
            f"({'above' if anomaly['direction'] == 'above' else 'below'} predicted)."
        )

    st.divider()

    # ── Key metric cards ──
    m1, m2, m3, m4 = st.columns(4)
    delta_val = r["user_rent"] - preds["predicted_rent"]

    with m1:
        st.metric(
            label="Ensemble Prediction",
            value=f"${preds['predicted_rent']:,.0f}",
            help="Inverse-MAPE weighted average of XGBoost + Prophet + LSTM",
        )
    with m2:
        st.metric(
            label="Your Rent",
            value=f"${r['user_rent']:,.0f}",
            delta=f"${delta_val:+,.0f} vs predicted",
            delta_color="inverse" if is_anomaly else "off",
        )
    with m3:
        st.metric(
            label="Anomaly Score",
            value=f"{anomaly['anomaly_score']:.2f}σ",
            delta="above threshold" if is_anomaly else "within threshold",
            delta_color="inverse" if is_anomaly else "off",
            help="Standard deviations from the predicted rent. >2.0σ is flagged as anomalous.",
        )
    with m4:
        st.metric(
            label="Data Confidence",
            value=conf.upper(),
            help=(
                "HIGH = historical date with exact training data. "
                "MEDIUM = future date, all live APIs succeeded. "
                "LOW = future date, some API fallbacks used."
            ),
        )

    st.divider()

    # ── Gauge chart ──
    st.subheader("Rent Position vs. Kalman Confidence Bounds")
    gauge_fig = _make_gauge(
        r["user_rent"], bounds["lower"], bounds["upper"],
        preds["predicted_rent"], is_anomaly,
    )
    st.pyplot(gauge_fig, use_container_width=True)
    plt.close(gauge_fig)

    st.divider()

    # ── Model breakdown ──
    st.subheader("Individual Model Predictions")

    model_rows = [
        ("XGBoost",  preds["xgb_pred"],      "4.92% CV MAPE", "Tabular features + neighborhood dummies",     ""),
        ("Prophet",  preds["prophet_pred"],   "4.45% CV MAPE", "Per-neighborhood trend + seasonality",        ""),
        ("LSTM",     preds["lstm_pred"],      "3.97% CV MAPE", "12-month sequence + neighborhood embeddings", ""),
        ("Ensemble", preds["predicted_rent"], "2.66% CV MAPE", "Inverse-MAPE weighted average",               "ensemble"),
    ]

    model_cols = st.columns(4)
    for col, (name, pred, mape, desc, extra_class) in zip(model_cols, model_rows):
        with col:
            diff      = r["user_rent"] - pred
            diff_pct  = diff / pred * 100
            diff_color = "#dc3545" if abs(diff_pct) > 5 else "#198754"
            diff_arrow = "↑" if diff > 0 else "↓"
            st.markdown(
                f"""
<div class="model-card {extra_class}">
    <div class="model-name">{name}</div>
    <div class="model-pred">${pred:,.0f}</div>
    <div class="model-diff" style="color:{diff_color};">
        {diff_arrow}&nbsp;${abs(diff):,.0f} ({abs(diff_pct):.1f}%) vs yours
    </div>
    <div class="model-meta">{mape}<br>{desc}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Feature detail expander ──
    with st.expander("📊 Features Used in This Prediction"):
        feat_labels = {
            "permits_trailing6": "Residential permits (trailing 6 months)",
            "employment_growth": "LA MSA employment growth (% MoM)",
            "median_income":     "Neighborhood median household income ($)",
            "renter_rate":       "Renter-occupied housing fraction",
            "covid":             "COVID indicator (1 = Mar 2020 – Jun 2021)",
            "month":             "Calendar month (1–12)",
            "rent_yoy_growth":   "Year-over-year rent growth (%)",
        }

        feat_rows = []
        for key, val in r["features"].items():
            feat_rows.append({
                "Feature":     key,
                "Value":       f"{val:.4g}" if isinstance(val, float) else str(val),
                "Description": feat_labels.get(key, ""),
            })

        feat_df = pd.DataFrame(feat_rows).set_index("Feature")
        st.dataframe(feat_df, use_container_width=True)


# ── Render: landing ────────────────────────────────────────────────────────────
else:
    st.markdown("### How It Works")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**1 · Enter Inputs**

Choose a neighborhood and month from the sidebar, then enter the rent price you want to check — a listing, your current rent, or any price you're curious about.
""")
    with c2:
        st.markdown("""
**2 · Run the Ensemble**

Three models predict what rent *should* be:
XGBoost (economic features), Prophet (trend + seasonality), and LSTM (temporal sequences with neighborhood embeddings). Predictions are combined with inverse-MAPE weights.
""")
    with c3:
        st.markdown("""
**3 · Check the Anomaly Flag**

A per-neighborhood Kalman Filter tracks the expected range dynamically — widening during volatile periods, narrowing during stable ones. Rents outside the 2σ interval are flagged for investigation.
""")

    st.divider()

    st.markdown("#### 22 Neighborhoods Covered")
    tier_cols = st.columns(4)
    for col, (tier, hoods) in zip(tier_cols, PRICE_TIERS.items()):
        with col:
            st.markdown(f"**{tier}**")
            for h in hoods:
                st.markdown(f"- {h}")

    st.divider()

    st.markdown("#### Model Performance (Walk-Forward CV, 2020–2025)")
    perf_data = {
        "Model":    ["XGBoost", "Prophet", "LSTM", "**Ensemble**"],
        "MAPE":     ["4.92%",   "4.45%",   "3.97%", "**2.66%**"],
        "R²":       ["—",       "—",       "0.975", "**0.982**"],
        "Approach": [
            "Tabular features, neighborhood dummies, no lag",
            "Logistic growth, additive seasonality (fourier_order=3)",
            "Global model, embedding_dim=8, hidden=128, seq_len=12",
            "Inverse-MAPE weights (XGB/Prophet/LSTM)",
        ],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)

    st.divider()
    st.info(
        "👈 **Select a neighborhood, date, and rent price in the sidebar, "
        "then click Analyze Rent** to run the full prediction pipeline."
    )

