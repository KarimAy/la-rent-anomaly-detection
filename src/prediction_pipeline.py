"""
Reusable prediction pipeline for LA Rent Anomaly Detection.
Exposes: load_models, get_features, predict_ensemble, get_kalman_bounds, detect_anomaly.
"""
import os
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import warnings
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data"

# ── Constants ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH   = 12
ANOMALY_THRESHOLD = 2.0

W_XGB     = 1 / 4.92
W_PROPHET = 1 / 4.45
W_LSTM    = 1 / 3.97
W_TOTAL   = W_XGB + W_PROPHET + W_LSTM

CONTINUOUS_FEATURES = [
    "permits_trailing6",
    "employment_growth",
    "median_income",
    "renter_rate",
    "covid",
    "month",
    "rent_yoy_growth",
]

NEIGHBORHOOD_ZIP_MAP = {
    "University Park": ["90007"],
    "Exposition Park": ["90037"],
    "Vermont Square":  ["90044"],
    "Koreatown":       ["90005", "90006", "90010"],
    "West Adams":      ["90016", "90018"],
    "Leimert Park":    ["90008"],
    "Boyle Heights":   ["90033", "90023"],
    "Silver Lake":     ["90026", "90039"],
    "Los Feliz":       ["90027"],
    "Highland Park":   ["90042"],
    "Glassell Park":   ["90065"],
    "Culver City":     ["90230", "90232"],
    "Mar Vista":       ["90066"],
    "Palms":           ["90034"],
    "West Hollywood":  ["90046", "90069"],
    "Mid-Wilshire":    ["90036"],
    "Hancock Park":    ["90004"],
    "Beverly Hills":   ["90210", "90211"],
    "Brentwood":       ["90049"],
    "Santa Monica":    ["90401", "90403", "90405"],
    "Westwood":        ["90024"],
    "Downtown LA":     ["90012", "90014", "90017"],
}

RESIDENTIAL_PERMIT_TYPES = [
    "Bldg-New", "Bldg-Addition", "Bldg-Alter/Repair", "Bldg-Demolition"
]
RESIDENTIAL_PERMIT_USES = [
    "Dwelling - Single Family", "Accessory Dwelling Unit",
    "Apartment", "Duplex", "Condominium",
]

HISTORICAL_CUTOFF = pd.Timestamp("2025-12-31")


# ── LSTM Architecture ─────────────────────────────────────────────────────────
class LSTMWithEmbedding(nn.Module):
    def __init__(self, num_neighborhoods, embedding_dim, input_size,
                 hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_neighborhoods, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=input_size + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_seq, neighborhood_id):
        emb = self.embedding(neighborhood_id)
        emb_expanded = emb.unsqueeze(1).expand(-1, x_seq.size(1), -1)
        x = torch.cat([x_seq, emb_expanded], dim=2)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out).squeeze()


# ── Public API ────────────────────────────────────────────────────────────────

def load_models() -> dict:
    """
    Load all saved models from models/.
    Returns a dict with keys: xgb, xgb_features, prophet, lstm, lstm_scaler,
    lstm_stats, lstm_idx, lstm_neighborhoods, kalman_last_state, device.
    """
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cpu")


    models = {}
    models["xgb"]               = joblib.load(MODELS_DIR / "xgb_final.pkl")
    models["xgb_features"]      = joblib.load(MODELS_DIR / "xgb_features.pkl")
    models["prophet"]           = joblib.load(MODELS_DIR / "prophet_final.pkl")
    models["lstm_scaler"]       = joblib.load(MODELS_DIR / "lstm_scaler.pkl")
    models["lstm_stats"]        = joblib.load(MODELS_DIR / "lstm_neighborhood_stats.pkl")
    models["lstm_idx"]          = joblib.load(MODELS_DIR / "lstm_neighborhood_idx.pkl")
    models["lstm_neighborhoods"] = joblib.load(MODELS_DIR / "lstm_neighborhoods.pkl")
    models["kalman_last_state"] = joblib.load(MODELS_DIR / "kalman_last_state.pkl")

    lstm = LSTMWithEmbedding(
        num_neighborhoods=22,
        embedding_dim=8,
        input_size=7,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
    ).to(device)
    lstm.load_state_dict(
        torch.load(MODELS_DIR / "lstm_final.pt", map_location=device, weights_only=True)
    )
    lstm.eval()

    models["lstm"]   = lstm
    models["device"] = device

    logger.info("All models loaded from %s", MODELS_DIR)
    return models


def get_features(neighborhood: str, date) -> tuple:
    """
    Returns (features_dict, confidence) where confidence is 'high', 'medium', or 'low'.

    Historical mode (date <= 2025-12-31): looks up exact row from master_model.csv.
    Future mode: fetches live data from LA City Open Data and BLS APIs, forward-fills
    demographics from last known ACS values.
    """
    date = pd.Timestamp(date)
    if date <= HISTORICAL_CUTOFF:
        return _get_historical_features(neighborhood, date)
    return _get_future_features(neighborhood, date)


def predict_ensemble(models: dict, features: dict, neighborhood: str, date) -> dict:
    """
    Run XGBoost, Prophet, and LSTM then combine with inverse-MAPE weights.

    Returns dict with keys: predicted_rent, xgb_pred, prophet_pred, lstm_pred.
    """
    date = pd.Timestamp(date)

    xgb_pred     = _predict_xgb(models, features, neighborhood)
    prophet_pred = _predict_prophet(models, neighborhood, date)
    lstm_pred    = _predict_lstm(models, neighborhood, date, xgb_pred, prophet_pred)

    ensemble_pred = (W_XGB * xgb_pred + W_PROPHET * prophet_pred + W_LSTM * lstm_pred) / W_TOTAL

    return {
        "predicted_rent": float(ensemble_pred),
        "xgb_pred":       float(xgb_pred),
        "prophet_pred":   float(prophet_pred),
        "lstm_pred":      float(lstm_pred),
    }


def get_kalman_bounds(models: dict, neighborhood: str, date, ensemble_pred: float) -> dict:
    """
    Returns dict with keys: lower, upper.

    Historical mode (date <= 2025-12-31): looks up from anomaly_results.csv.
    Future mode: derives bounds from kalman_last_state.pkl using 2.0 SD threshold.
    """
    date = pd.Timestamp(date)
    if date <= HISTORICAL_CUTOFF:
        return _get_historical_bounds(neighborhood, date)
    return _get_future_bounds(models, neighborhood, ensemble_pred)


def detect_anomaly(user_rent: float, lower: float, upper: float, ensemble_pred: float) -> dict:
    """
    Returns is_anomaly (bool), anomaly_score (float — SDs away from ensemble),
    and direction ('above' or 'below' relative to ensemble).
    """
    smoothed_std  = (upper - lower) / (2 * ANOMALY_THRESHOLD)
    is_anomaly    = user_rent > upper or user_rent < lower
    anomaly_score = abs(user_rent - ensemble_pred) / (smoothed_std + 1e-8)
    direction     = "above" if user_rent >= ensemble_pred else "below"

    return {
        "is_anomaly":    bool(is_anomaly),
        "anomaly_score": float(anomaly_score),
        "direction":     direction,
    }


# ── Historical helpers ────────────────────────────────────────────────────────

def _get_historical_features(neighborhood: str, date: pd.Timestamp) -> tuple:
    master = _load_master()

    row = master[
        (master["Neighborhood"] == neighborhood) &
        (master["Date"] == date)
    ]

    if row.empty:
        # Relax to same year-month (handles slight day mismatches)
        row = master[
            (master["Neighborhood"] == neighborhood) &
            (master["Date"].dt.year == date.year) &
            (master["Date"].dt.month == date.month)
        ]

    if row.empty:
        raise ValueError(
            f"No historical features for {neighborhood} on {date.date()}. "
            "Check that the date falls within 2020-01 to 2025-12 and the "
            "neighborhood name matches exactly."
        )

    features = {col: float(row.iloc[0][col]) for col in CONTINUOUS_FEATURES}
    return features, "high"


def _get_historical_bounds(neighborhood: str, date: pd.Timestamp) -> dict:
    anomaly_df = pd.read_csv(DATA_DIR / "model_csv" / "anomaly_results.csv")
    anomaly_df["Date"] = pd.to_datetime(anomaly_df["Date"])

    row = anomaly_df[
        (anomaly_df["Neighborhood"] == neighborhood) &
        (anomaly_df["Date"] == date)
    ]

    if row.empty:
        row = anomaly_df[
            (anomaly_df["Neighborhood"] == neighborhood) &
            (anomaly_df["Date"].dt.year == date.year) &
            (anomaly_df["Date"].dt.month == date.month)
        ]

    if row.empty:
        raise ValueError(
            f"No Kalman bounds for {neighborhood} on {date.date()} in anomaly_results.csv."
        )

    return {
        "lower": float(row.iloc[0]["lower_bound"]),
        "upper": float(row.iloc[0]["upper_bound"]),
    }


# ── Future helpers ────────────────────────────────────────────────────────────

def _get_future_features(neighborhood: str, date: pd.Timestamp) -> tuple:
    confidence = "medium"
    features   = {}

    # permits_trailing6 — LA City Open Data API
    try:
        features["permits_trailing6"] = _fetch_permits_trailing6(neighborhood, date)
    except Exception as exc:
        logger.warning("Permits API failed (%s). Defaulting to 0.", exc)
        features["permits_trailing6"] = 0.0
        confidence = "low"

    # employment_growth — BLS API
    try:
        features["employment_growth"] = _fetch_bls_employment_growth(date)
    except Exception as exc:
        logger.warning("BLS API failed (%s). Forward-filling last known value.", exc)
        features["employment_growth"] = _last_known_employment_growth()
        confidence = "low"

    # demographics — forward-fill from last known ACS (2023)
    # CENSUS_API_KEY = os.environ.get("CENSUS_API_KEY", "") reserved for live ACS fetch
    try:
        demo = _last_known_demographics(neighborhood)
        features["median_income"] = demo["median_income"]
        features["renter_rate"]   = demo["renter_rate"]
    except Exception as exc:
        logger.warning("Demographics lookup failed (%s).", exc)
        features["median_income"] = np.nan
        features["renter_rate"]   = np.nan
        confidence = "low"

    # derived
    features["covid"] = 0
    features["month"] = date.month

    try:
        features["rent_yoy_growth"] = _last_known_rent_yoy_growth(neighborhood)
    except Exception as exc:
        logger.warning("rent_yoy_growth lookup failed (%s). Using 0.", exc)
        features["rent_yoy_growth"] = 0.0
        confidence = "low"

    return features, confidence


def _fetch_permits_trailing6(neighborhood: str, date: pd.Timestamp) -> float:
    """Count residential permits issued in the 6 months ending at date."""
    zips = NEIGHBORHOOD_ZIP_MAP.get(neighborhood, [])
    if not zips:
        return 0.0

    start = (date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    end   = date.strftime("%Y-%m-%d")

    url    = "https://data.lacity.org/resource/pi9x-tg5x.csv"
    params = {
        "$limit":  10000,
        "$where":  f"issue_date >= '{start}' AND issue_date <= '{end}'",
        "$select": "permit_nbr,zip_code,permit_type,use_desc",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))
    if df.empty:
        return 0.0

    df["zip_code"] = (
        df["zip_code"].astype(str)
        .str.replace(".0", "", regex=False)
        .str.strip()
        .str.zfill(5)
    )

    mask = (
        df["zip_code"].isin(zips) &
        df["permit_type"].isin(RESIDENTIAL_PERMIT_TYPES) &
        df["use_desc"].isin(RESIDENTIAL_PERMIT_USES)
    )
    return float(mask.sum())


def _fetch_bls_employment_growth(date: pd.Timestamp) -> float:
    """Fetch month-over-month employment growth % from BLS for the given month."""
    api_key = os.environ.get("BLS_API_KEY", "")

    # Pull prior year + target year to compute MoM change
    payload = {
        "seriesid":        ["SMU06310800000000001"],
        "startyear":       str(date.year - 1),
        "endyear":         str(date.year),
        "registrationkey": api_key,
    }

    resp = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    if data["status"] != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API: {data.get('message', 'unknown error')}")

    series = data["Results"]["series"][0]["data"]
    emp_df = pd.DataFrame(series)
    emp_df["value"] = pd.to_numeric(emp_df["value"], errors="coerce")
    emp_df["Date"]  = (
        pd.to_datetime(
            emp_df["year"] + "-" + emp_df["period"].str.replace("M", ""),
            format="%Y-%m",
        )
        + pd.offsets.MonthEnd(0)
    )
    emp_df = emp_df.sort_values("Date")
    emp_df["employment_growth"] = emp_df["value"].pct_change() * 100

    target = pd.Timestamp(year=date.year, month=date.month, day=1) + pd.offsets.MonthEnd(0)
    row    = emp_df[emp_df["Date"] == target]
    if row.empty:
        row = emp_df.dropna(subset=["employment_growth"]).tail(1)
    if row.empty:
        raise RuntimeError("No employment data returned by BLS API.")

    return float(row["employment_growth"].iloc[-1])


def _last_known_employment_growth() -> float:
    emp = pd.read_csv(DATA_DIR / "processed" / "employment_clean.csv")
    return float(emp.dropna(subset=["employment_growth"])["employment_growth"].iloc[-1])


def _last_known_demographics(neighborhood: str) -> dict:
    master = _load_master()
    nd = master[master["Neighborhood"] == neighborhood].dropna(
        subset=["median_income", "renter_rate"]
    )
    if nd.empty:
        raise ValueError(f"No demographic data for {neighborhood}")
    last = nd.iloc[-1]
    return {"median_income": float(last["median_income"]), "renter_rate": float(last["renter_rate"])}


def _last_known_rent_yoy_growth(neighborhood: str) -> float:
    master = _load_master()
    nd = master[
        (master["Neighborhood"] == neighborhood) &
        master["rent_yoy_growth"].notna()
    ]
    if nd.empty:
        return 0.0
    return float(nd["rent_yoy_growth"].iloc[-1])


def _get_future_bounds(models: dict, neighborhood: str, ensemble_pred: float) -> dict:
    kalman_df = models["kalman_last_state"]
    row = kalman_df[kalman_df["Neighborhood"] == neighborhood]

    if row.empty:
        raise ValueError(f"No Kalman last state for {neighborhood}")

    smoothed_std = float(row["smoothed_std"].iloc[0])
    return {
        "lower": ensemble_pred - ANOMALY_THRESHOLD * smoothed_std,
        "upper": ensemble_pred + ANOMALY_THRESHOLD * smoothed_std,
    }


# ── Model prediction helpers ──────────────────────────────────────────────────

def _predict_xgb(models: dict, features: dict, neighborhood: str) -> float:
    """One-hot encode neighborhood (Beverly Hills = reference = all zeros) and predict."""
    xgb_features = models["xgb_features"]

    row = {f: 0 for f in xgb_features}

    dummy = f"Neighborhood_{neighborhood}"
    if dummy in row:
        row[dummy] = 1

    for feat in CONTINUOUS_FEATURES:
        row[feat] = features.get(feat, 0.0)

    X = pd.DataFrame([row])[xgb_features]
    return float(models["xgb"].predict(X)[0])


def _predict_prophet(models: dict, neighborhood: str, date: pd.Timestamp) -> float:
    """Prophet logistic growth prediction with saved floor/cap bounds."""
    info = models["prophet"][neighborhood]

    future_df = pd.DataFrame({
        "ds":    [date],
        "floor": [info["floor"]],
        "cap":   [info["cap"]],
    })

    forecast = info["model"].predict(future_df)
    pred = float(forecast["yhat"].values[0])
    # Clip to logistic bounds
    return max(info["floor"], min(info["cap"], pred))


def _predict_lstm(
    models: dict,
    neighborhood: str,
    date: pd.Timestamp,
    xgb_pred: float,
    prophet_pred: float,
) -> float:
    """
    LSTM prediction using a 12-month lookback sequence from master_model.csv.
    Falls back to the XGBoost+Prophet average when fewer than 12 months are available.
    """
    master = _load_master()
    nd = (
        master[master["Neighborhood"] == neighborhood]
        .sort_values("Date")
    )
    nd = nd[nd["Date"] <= date].tail(SEQUENCE_LENGTH)

    if len(nd) < SEQUENCE_LENGTH:
        logger.info(
            "LSTM skipped for %s %s — %d months history (need %d). "
            "Using XGBoost+Prophet average.",
            neighborhood, date.date(), len(nd), SEQUENCE_LENGTH,
        )
        return (xgb_pred + prophet_pred) / 2.0

    seq       = models["lstm_scaler"].transform(nd[CONTINUOUS_FEATURES].values)
    seq_t     = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(models["device"])
    nd_id     = torch.tensor(
        [models["lstm_idx"][neighborhood]], dtype=torch.long
    ).to(models["device"])

    with torch.no_grad():
        pred_norm = float(models["lstm"](seq_t, nd_id).cpu().numpy())

    stats  = models["lstm_stats"]
    n_row  = stats[stats["Neighborhood"] == neighborhood].iloc[0]
    return float(pred_norm * n_row["rent_std"] + n_row["rent_mean"])


# ── Shared data loader (cached via module-level singleton) ────────────────────

_master_cache: pd.DataFrame | None = None


def _load_master() -> pd.DataFrame:
    global _master_cache
    if _master_cache is None:
        _master_cache = pd.read_csv(DATA_DIR / "processed" / "master_model.csv")
        _master_cache["Date"] = pd.to_datetime(_master_cache["Date"])
    return _master_cache
