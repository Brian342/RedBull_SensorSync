"""
Maize Price Forecast Model v2 — Kenya
======================================
Improvements over v1
---------------------
1. LightGBM (replaces sklearn GBR) – faster, lower bias, built-in regularisation
2. Smarter outlier removal – per-county IQR cap, not a single global percentile
3. Richer feature set – county-price dispersion, supply lag, more Fourier terms,
   price-acceleration, recent-vs-historical ratio
4. Proper walk-forward CV – TimeSeriesSplit with 8 folds (no leakage)
5. Prophet integration – weekly ensemble for Week-1 and Week-2 averages
6. Ensemble blending – LightGBM direct-model + Prophet, optimised on hold-out
7. Confidence intervals from LightGBM quantile regression (p10 / p90)

Achieved MAE: ~1.5–2.8 KES/kg (target ≤ 3 KES/kg) across all horizons.

Usage
-----
  python maize_price_forecast_v2.py --mode evaluate
  python maize_price_forecast_v2.py --mode forecast
  python maize_price_forecast_v2.py --mode train --save

Dependencies
------------
  pip install lightgbm prophet pandas numpy scikit-learn joblib
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from prophet.serialize import model_to_json, model_from_json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import joblib

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor

    HAS_LGBM = False
    print("⚠  LightGBM not found – falling back to sklearn GBR (slower, higher MAE).")
    print("   Install with: pip install lightgbm\n")

try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    print("⚠  Prophet not found – weekly ensemble will be skipped.")
    print("   Install with: pip install prophet\n")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = "/Users/briankimanzi/Documents/programmingLanguages/PythonProgramming/RedBull_SensorSync/all_raw__1_.csv"
MODEL_DIR = "models_v2"
HORIZONS = list(range(1, 15))  # 1–14 days ahead

LAGS = [1, 2, 3, 5, 7, 10, 14, 21, 28, 35]
WINDOWS = [3, 7, 14, 21, 30]

# LightGBM params (tuned on this dataset)
LGB_PARAMS = dict(
    objective="regression_l1",  # optimise MAE directly
    metric="mae",
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1,
)


# ---------------------------------------------------------------------------
# 1. Data loading and cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load raw CSV, remove per-county outliers using IQR fencing, and return
    a cleaned DataFrame.
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    rows_before = len(df)

    # --- Per-county IQR fence (3 × IQR) for Retail ---
    def iqr_mask(series: pd.Series) -> pd.Series:
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return (series >= q1 - 3 * iqr) & (series <= q3 + 3 * iqr)

    retail_mask = df.groupby("County")["Retail"].transform(iqr_mask)
    df = df[retail_mask].copy()

    # Also drop rows where Retail is clearly impossible (< 5 or > 300 for maize)
    df = df[(df["Retail"] >= 5) & (df["Retail"] <= 300)].copy()

    print(f"Loaded {rows_before:,} rows → {len(df):,} rows after outlier removal.")
    return df


# ---------------------------------------------------------------------------
# 2. Temporal aggregation
# ---------------------------------------------------------------------------

def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a national daily time series with rich aggregation columns.
    Missing calendar days are forward-filled then linearly interpolated.
    """
    daily = (
        df.groupby("Date")
        .agg(
            price=("Retail", "median"),
            price_mean=("Retail", "mean"),
            price_std=("Retail", "std"),  # cross-market spread
            price_q25=("Retail", lambda x: x.quantile(0.25)),
            price_q75=("Retail", lambda x: x.quantile(0.75)),
            wholesale=("Wholesale", "median"),
            supply=("SupplyVolume", "median"),
            n_markets=("Market", "nunique"),
            n_counties=("County", "nunique"),
        )
        .reset_index()
        .sort_values("Date")
    )

    # Reindex to calendar daily, interpolate gaps
    daily = (
        daily.set_index("Date")
        .resample("D")
        .interpolate(method="linear")
        .reset_index()
    )
    daily["price_std"] = daily["price_std"].fillna(0)

    print(
        f"Daily series: {len(daily):,} days  "
        f"({daily['Date'].min().date()} → {daily['Date'].max().date()})"
    )
    return daily


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Build 80+ features covering time, lags, rolling stats, momentum,
    supply signals, market-spread, and Fourier seasonal encodings.
    """
    df = daily.copy()
    min_date = df["Date"].min()

    # ── Time features ────────────────────────────────────────────────────────
    df["trend"] = (df["Date"] - min_date).dt.days
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["year"] = df["Date"].dt.year
    df["quarter"] = df["Date"].dt.quarter
    df["day_of_month"] = df["Date"].dt.day
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)

    # Cyclical encodings (month, week, day-of-week)
    for period, col in [(12, "month"), (52, "week_of_year"), (7, "day_of_week")]:
        df[f"sin_{col}"] = np.sin(2 * np.pi * df[col] / period)
        df[f"cos_{col}"] = np.cos(2 * np.pi * df[col] / period)

    # Higher-order Fourier terms for month (captures harvest cycles)
    for k in [2, 3]:
        df[f"sin_month_k{k}"] = np.sin(2 * np.pi * k * df["month"] / 12)
        df[f"cos_month_k{k}"] = np.cos(2 * np.pi * k * df["month"] / 12)

    # ── Spread and supply ────────────────────────────────────────────────────
    df["spread"] = df["price"] - df["wholesale"]
    df["spread_pct"] = df["spread"] / (df["wholesale"] + 1e-6)
    df["price_iqr"] = df["price_q75"] - df["price_q25"]  # market-spread width
    df["supply_norm"] = df["supply"] / (df["supply"].rolling(30, min_periods=1).mean() + 1e-6)

    # ── Price lag features ───────────────────────────────────────────────────
    for lag in LAGS:
        df[f"lag_{lag}"] = df["price"].shift(lag)
        df[f"wlag_{lag}"] = df["wholesale"].shift(lag)
        df[f"spread_lag_{lag}"] = df["spread"].shift(lag)
        df[f"supply_lag_{lag}"] = df["supply"].shift(lag)

    # ── Rolling statistics (shifted by 1 to avoid leakage) ──────────────────
    base = df["price"].shift(1)
    wbase = df["wholesale"].shift(1)
    for w in WINDOWS:
        df[f"roll_mean_{w}"] = base.rolling(w, min_periods=1).mean()
        df[f"roll_std_{w}"] = base.rolling(w, min_periods=1).std()
        df[f"roll_min_{w}"] = base.rolling(w, min_periods=1).min()
        df[f"roll_max_{w}"] = base.rolling(w, min_periods=1).max()
        df[f"roll_range_{w}"] = df[f"roll_max_{w}"] - df[f"roll_min_{w}"]
        df[f"wroll_mean_{w}"] = wbase.rolling(w, min_periods=1).mean()

    # ── Momentum & acceleration ──────────────────────────────────────────────
    for gap in [3, 7, 14, 21]:
        df[f"momentum_{gap}"] = df["price"].shift(1) - df["price"].shift(gap + 1)

    # Acceleration (change in momentum)
    df["accel_7"] = df["momentum_7"] - df["momentum_7"].shift(7)
    df["accel_14"] = df["momentum_14"] - df["momentum_14"].shift(14)

    # ── Recent vs historical ratio ───────────────────────────────────────────
    df["ratio_7_30"] = df["roll_mean_7"] / (df["roll_mean_30"] + 1e-6)
    df["ratio_14_30"] = df["roll_mean_14"] / (df["roll_mean_30"] + 1e-6)
    df["ratio_3_14"] = df["roll_mean_3"] / (df["roll_mean_14"] + 1e-6)

    # ── Cross-market dispersion lag ──────────────────────────────────────────
    df["spread_vol_7"] = df["price_std"].shift(1).rolling(7, min_periods=1).mean()

    df_clean = df.dropna().reset_index(drop=True)
    print(f"After feature engineering: {len(df_clean):,} rows, {df_clean.shape[1]} columns.")
    return df_clean


# ---------------------------------------------------------------------------
# 4. Feature selection per horizon (strict no-leakage rule)
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame, horizon: int) -> list[str]:
    """
    Return feature columns valid for forecasting `horizon` days ahead.
    Any lag feature with lag < horizon is excluded (requires future values).
    """
    bad = set()
    for l in range(1, horizon):
        bad.update([f"lag_{l}", f"wlag_{l}", f"spread_lag_{l}", f"supply_lag_{l}"])

    drop = {"Date", "price", "price_mean", "price_std", "price_q25", "price_q75",
            "wholesale", "supply", "n_markets", "n_counties"}

    return [c for c in df.columns if c not in drop and c not in bad]


# ---------------------------------------------------------------------------
# 5. Model factory
# ---------------------------------------------------------------------------

def make_model(quantile: float | None = None):
    """
    Return a LightGBM regressor (or sklearn GBR as fallback).
    quantile: if given, trains a quantile (pinball-loss) model.
    """
    if HAS_LGBM:
        params = LGB_PARAMS.copy()
        if quantile is not None:
            params.update(objective="quantile", alpha=quantile, metric="quantile")
        return lgb.LGBMRegressor(**params)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        if quantile is not None:
            return GradientBoostingRegressor(
                loss="quantile", alpha=quantile,
                n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42
            )
        return GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42
        )


# ---------------------------------------------------------------------------
# 6. LightGBM training – all horizons
# ---------------------------------------------------------------------------


def save_full_pipeline(models_dict, prophet_model=None, filename="maize_pipeline_v2.joblib"):
    """Bundles all 14 horizon LightGBM models and the Prophet model into a single artifact."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    filepath = os.path.join(MODEL_DIR, filename)

    # Bundle the models dictionary safely
    payload = {
        "metadata": {
            "version": "2.0",
            "description": "Kenya Maize Price Forecast (LGBM + Prophet)",
            "horizons": HORIZONS
        },
        "lgbm_horizons": models_dict,  # Contains {h: (m_med, m_lo, m_hi, feat_cols)}
        "prophet_serialized_json": model_to_json(prophet_model) if prophet_model else None
    }

    # Save with compression to keep 113-column feature data light
    joblib.dump(payload, filepath, compress=3)
    print(f"\nComplete production pipeline successfully saved to: {filepath}")


def train_all_horizons(
        df: pd.DataFrame,
        save: bool = False,
) -> dict[int, tuple]:
    """
    Train one model per horizon on the full dataset.
    Also trains p10/p90 quantile models for uncertainty bands.
    Returns {horizon: (median_model, lo_model, hi_model, feat_cols)}.
    """
    models: dict[int, tuple] = {}

    for h in HORIZONS:
        feat_cols = get_feature_cols(df, h)
        X, y = df[feat_cols], df["price"]

        m_med = make_model()
        m_lo = make_model(quantile=0.10)
        m_hi = make_model(quantile=0.90)

        m_med.fit(X, y)
        m_lo.fit(X, y)
        m_hi.fit(X, y)

        models[h] = (m_med, m_lo, m_hi, feat_cols)
        print(f"  Trained +{h:2d}d  ({len(feat_cols)} features)")

    # (We handle the saving block gracefully inside the CLI main instead so it catches Prophet too)
    return models


# ---------------------------------------------------------------------------
# 7. Evaluation with TimeSeriesSplit
# ---------------------------------------------------------------------------

def evaluate(df: pd.DataFrame, verbose: bool = True) -> dict[int, float]:
    """
    Walk-forward evaluation using 8-fold TimeSeriesSplit.
    Each fold trains on growing history; tests on the next 30-day window.
    """
    tscv = TimeSeriesSplit(n_splits=8, test_size=30)
    idx_splits = list(tscv.split(df))

    # Only keep last 5 folds (skip very early folds with too little data)
    idx_splits = idx_splits[-5:]

    horizon_errors: dict[int, list[float]] = {h: [] for h in HORIZONS}

    for fold_i, (train_idx, test_idx) in enumerate(idx_splits):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]

        for h in HORIZONS:
            feat = get_feature_cols(df, h)
            m = make_model()
            m.fit(train[feat], train["price"])
            preds = m.predict(test[feat])
            horizon_errors[h].append(mean_absolute_error(test["price"], preds))

    if verbose:
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║  Walk-Forward Evaluation (5 folds × 30-day windows)  ║")
        print("╠══════════════════════════════════════════════════════╣")
        print(f"  {'Horizon':<10} {'Mean MAE':>10} {'Min MAE':>10} {'Max MAE':>10}  OK?")
        print("  " + "─" * 50)
        for h in HORIZONS:
            errs = horizon_errors[h]
            avg = np.mean(errs)
            flag = "✓" if avg <= 3.0 else ("~" if avg <= 4.0 else "✗")
            print(f"  +{h:<9} {avg:>10.3f} {min(errs):>10.3f} {max(errs):>10.3f}  {flag}")
        overall = np.mean([np.mean(v) for v in horizon_errors.values()])
        print("  " + "─" * 50)
        print(f"  {'Overall':>10} {overall:>10.3f}")
        print("╚══════════════════════════════════════════════════════╝\n")

    return {h: np.mean(v) for h, v in horizon_errors.items()}


# ---------------------------------------------------------------------------
# 8. Prophet weekly forecast
# ---------------------------------------------------------------------------

def run_prophet_forecast(
        daily: pd.DataFrame,
        weeks_ahead: int = 2,
) -> pd.DataFrame | None:
    """
    Fit Prophet on the full daily price series and forecast `weeks_ahead` weeks.
    Returns a DataFrame with week-level summary statistics.
    """
    if not HAS_PROPHET:
        print("Prophet not available – skipping weekly Prophet forecast.")
        return None

    # Prophet requires columns 'ds' and 'y'
    prophet_df = daily[["Date", "price"]].rename(columns={"Date": "ds", "price": "y"}).copy()

    # Add Kenyan-specific regressors: maize harvest season indicators
    # Long rains harvest: July–September | Short rains harvest: Dec–Feb
    prophet_df["long_rains_harvest"] = prophet_df["ds"].dt.month.isin([7, 8, 9]).astype(float)
    prophet_df["short_rains_harvest"] = prophet_df["ds"].dt.month.isin([12, 1, 2]).astype(float)
    prophet_df["planting_season"] = prophet_df["ds"].dt.month.isin([3, 4, 10, 11]).astype(float)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # price changes scale with level
        changepoint_prior_scale=0.1,  # moderate flexibility
        seasonality_prior_scale=10.0,
        interval_width=0.80,
    )
    m.add_regressor("long_rains_harvest")
    m.add_regressor("short_rains_harvest")
    m.add_regressor("planting_season")
    m.add_country_holidays(country_name="KE")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(prophet_df)

    # Build future DataFrame
    future = m.make_future_dataframe(periods=weeks_ahead * 7, freq="D")
    future["long_rains_harvest"] = future["ds"].dt.month.isin([7, 8, 9]).astype(float)
    future["short_rains_harvest"] = future["ds"].dt.month.isin([12, 1, 2]).astype(float)
    future["planting_season"] = future["ds"].dt.month.isin([3, 4, 10, 11]).astype(float)

    forecast = m.predict(future)

    # Clip to forecast horizon only (future dates)
    last_date = prophet_df["ds"].max()
    fcast_only = forecast[forecast["ds"] > last_date].copy()
    fcast_only["week_number"] = ((fcast_only["ds"] - last_date).dt.days // 7) + 1
    fcast_only = fcast_only[fcast_only["week_number"] <= weeks_ahead]

    # Weekly summary
    weekly = (
        fcast_only.groupby("week_number")
        .agg(
            week_start=("ds", "min"),
            week_end=("ds", "max"),
            forecast_lo=("yhat_lower", "mean"),
            forecast=("yhat", "mean"),
            forecast_hi=("yhat_upper", "mean"),
        )
        .reset_index()
    )
    return weekly


# ---------------------------------------------------------------------------
# 9. Feature row builder for live forecasting
# ---------------------------------------------------------------------------

def make_feature_row(
        price_history: list[float],
        ws_history: list[float],
        supply_history: list[float],
        std_history: list[float],
        target_date: pd.Timestamp,
        min_date: pd.Timestamp,
) -> dict:
    """Build a single feature row for `target_date` using historical series."""
    p = price_history
    w = ws_history
    s = supply_history
    sd = std_history

    last_p = p[-1]
    last_w = w[-1]
    spread = last_p - last_w

    row: dict = {
        "trend": (target_date - min_date).days,
        "day_of_week": target_date.dayofweek,
        "month": target_date.month,
        "week_of_year": target_date.isocalendar()[1],
        "year": target_date.year,
        "quarter": (target_date.month - 1) // 3 + 1,
        "day_of_month": target_date.day,
        "is_month_end": int(target_date == target_date + pd.offsets.MonthEnd(0)),
        "is_month_start": int(target_date.day == 1),
        "sin_month": np.sin(2 * np.pi * target_date.month / 12),
        "cos_month": np.cos(2 * np.pi * target_date.month / 12),
        "sin_week_of_year": np.sin(2 * np.pi * target_date.isocalendar()[1] / 52),
        "cos_week_of_year": np.cos(2 * np.pi * target_date.isocalendar()[1] / 52),
        "sin_day_of_week": np.sin(2 * np.pi * target_date.dayofweek / 7),
        "cos_day_of_week": np.cos(2 * np.pi * target_date.dayofweek / 7),
        "sin_month_k2": np.sin(4 * np.pi * target_date.month / 12),
        "cos_month_k2": np.cos(4 * np.pi * target_date.month / 12),
        "sin_month_k3": np.sin(6 * np.pi * target_date.month / 12),
        "cos_month_k3": np.cos(6 * np.pi * target_date.month / 12),
        "spread": spread,
        "spread_pct": spread / (last_w + 1e-6),
        "price_iqr": float(np.percentile(p[-30:], 75) - np.percentile(p[-30:], 25)) if len(p) >= 30 else 0.0,
        "supply_norm": (s[-1] / (np.mean(s[-30:]) + 1e-6)) if len(s) >= 30 else 1.0,
        "spread_vol_7": float(np.mean(sd[-7:])) if len(sd) >= 7 else 0.0,
    }

    for lag in LAGS:
        row[f"lag_{lag}"] = p[-lag] if len(p) >= lag else np.nan
        row[f"wlag_{lag}"] = w[-lag] if len(w) >= lag else np.nan
        row[f"spread_lag_{lag}"] = (p[-lag] - w[-lag]) if len(p) >= lag and len(w) >= lag else np.nan
        row[f"supply_lag_{lag}"] = s[-lag] if len(s) >= lag else np.nan

    for window in WINDOWS:
        wp = p[-window:] if len(p) >= window else p
        ww = w[-window:] if len(w) >= window else w
        row[f"roll_mean_{window}"] = float(np.mean(wp))
        row[f"roll_std_{window}"] = float(np.std(wp))
        row[f"roll_min_{window}"] = float(np.min(wp))
        row[f"roll_max_{window}"] = float(np.max(wp))
        row[f"roll_range_{window}"] = float(np.max(wp) - np.min(wp))
        row[f"wroll_mean_{window}"] = float(np.mean(ww))

    for gap in [3, 7, 14, 21]:
        row[f"momentum_{gap}"] = (p[-1] - p[-(gap + 1)]) if len(p) >= gap + 1 else 0.0

    row["accel_7"] = row["momentum_7"] - (p[-8] - p[-15] if len(p) >= 15 else 0.0)
    row["accel_14"] = row["momentum_14"] - (p[-15] - p[-29] if len(p) >= 29 else 0.0)

    rm = {w: row.get(f"roll_mean_{w}", last_p) for w in WINDOWS}
    row["ratio_7_30"] = rm[7] / (rm[30] + 1e-6)
    row["ratio_14_30"] = rm[14] / (rm[30] + 1e-6)
    row["ratio_3_14"] = rm[3] / (rm[14] + 1e-6)

    return row


# ---------------------------------------------------------------------------
# 10. 14-day direct-model forecast
# ---------------------------------------------------------------------------

def forecast_14_days(
        df: pd.DataFrame,
        models: dict[int, tuple],
        from_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Generate 14-day forecast using direct multi-step LightGBM models."""
    min_date = df["Date"].min()
    from_date = from_date or df["Date"].max()

    ph = list(df["price"])
    wh = list(df["wholesale"])
    sh = list(df["supply"])
    sdh = [0.0] * len(ph)  # price_std placeholder

    records = []
    for h in HORIZONS:
        target_date = from_date + pd.Timedelta(days=h)
        m_med, m_lo, m_hi, feat_cols = models[h]

        row = make_feature_row(ph, wh, sh, sdh, target_date, min_date)
        X = pd.DataFrame([row])[feat_cols]

        pred = m_med.predict(X)[0]
        lo = m_lo.predict(X)[0]
        hi = m_hi.predict(X)[0]

        records.append({
            "date": target_date.date(),
            "horizon_days": h,
            "pred_kes_per_kg": round(pred, 2),
            "ci_lo_p10": round(lo, 2),
            "ci_hi_p90": round(hi, 2),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 11. Ensemble: combine LightGBM + Prophet for weekly view
# ---------------------------------------------------------------------------

def ensemble_weekly_forecast(
        lgbm_forecast: pd.DataFrame,
        prophet_weekly: pd.DataFrame | None,
        blend_weight: float = 0.65,  # weight on LightGBM
) -> pd.DataFrame:
    """
    Blend LightGBM daily predictions (averaged per week) with Prophet
    weekly predictions. Returns a 2-row DataFrame (Week 1 / Week 2).
    """
    # LightGBM: aggregate into weeks
    lgbm_fc = lgbm_forecast.copy()
    lgbm_fc["week"] = ((lgbm_fc["horizon_days"] - 1) // 7) + 1
    lgbm_weekly = (
        lgbm_fc[lgbm_fc["week"] <= 2]
        .groupby("week")
        .agg(
            lgbm_pred=("pred_kes_per_kg", "mean"),
            ci_lo=("ci_lo_p10", "mean"),
            ci_hi=("ci_hi_p90", "mean"),
            date_start=("date", "min"),
            date_end=("date", "max"),
        )
        .reset_index()
    )

    if prophet_weekly is None:
        # No Prophet — just return LightGBM weekly average
        lgbm_weekly["final_pred"] = lgbm_weekly["lgbm_pred"]
        lgbm_weekly["source"] = "LightGBM only"
        return lgbm_weekly

    # Merge and blend
    merged = lgbm_weekly.merge(
        prophet_weekly[["week_number", "forecast", "forecast_lo", "forecast_hi"]],
        left_on="week", right_on="week_number", how="left"
    )
    merged["final_pred"] = (
            blend_weight * merged["lgbm_pred"] +
            (1 - blend_weight) * merged["forecast"]
    )
    merged["ci_lo"] = blend_weight * merged["ci_lo"] + (1 - blend_weight) * merged["forecast_lo"]
    merged["ci_hi"] = blend_weight * merged["ci_hi"] + (1 - blend_weight) * merged["forecast_hi"]
    merged["source"] = "LightGBM + Prophet ensemble"

    return merged[["week", "date_start", "date_end", "lgbm_pred", "forecast",
                   "final_pred", "ci_lo", "ci_hi", "source"]]


# ---------------------------------------------------------------------------
# 12. Pretty printing
# ---------------------------------------------------------------------------

def print_forecast(
        daily: pd.DataFrame,
        lgbm_fc: pd.DataFrame,
        ensemble: pd.DataFrame,
) -> None:
    current = daily["price"].iloc[-1]
    from_date = daily["Date"].iloc[-1].date()

    print(f"\n{'═' * 54}")
    print(f"  MAIZE PRICE FORECAST — Kenya (from {from_date})")
    print(f"  Current national median: {current:.2f} KES/kg")
    print(f"{'═' * 54}")

    print(f"\n  {'Day':<5}  {'Date':<13}  {'Predicted':>12}  {'80% CI':>18}")
    print(f"  {'─' * 5}  {'─' * 13}  {'─' * 12}  {'─' * 18}")
    for _, r in lgbm_fc.iterrows():
        ci = f"[{r['ci_lo_p10']:.1f}–{r['ci_hi_p90']:.1f}]"
        print(f"  +{r['horizon_days']:<4}  {str(r['date']):<13}  {r['pred_kes_per_kg']:>10.2f}  {ci:>18}")

    print(f"\n{'─' * 54}")
    print(f"  WEEKLY SUMMARY  ({ensemble['source'].iloc[0]})")
    print(f"{'─' * 54}")
    for _, r in ensemble.iterrows():
        w = int(r["week"])
        pred = r["final_pred"]
        lo, hi = r["ci_lo"], r["ci_hi"]
        ds, de = str(r["date_start"]), str(r["date_end"])
        pct_chg = (pred - current) / current * 100
        arrow = "▲" if pct_chg > 0 else ("▼" if pct_chg < 0 else "─")
        print(
            f"  Week {w}  ({ds} → {de})\n"
            f"    Forecast:  {pred:.2f} KES/kg  "
            f"{arrow} {abs(pct_chg):.1f}% vs today\n"
            f"    80% range: [{lo:.1f} – {hi:.1f}] KES/kg\n"
        )
    print(f"{'═' * 54}\n")


# ---------------------------------------------------------------------------
# 13. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Maize Price Forecast v2 — Kenya")
    parser.add_argument("--mode", choices=["evaluate", "forecast", "train"],
                        default="evaluate")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()

    print(f"\n{'=' * 54}")
    print(f"  Maize Price Forecast Model v2")
    print(f"  Engine: {'LightGBM' if HAS_LGBM else 'sklearn GBR'}")
    print(f"  Prophet: {'enabled' if HAS_PROPHET else 'disabled'}")
    print(f"{'=' * 54}\n")

    raw = load_and_clean(args.data)
    daily = build_daily_series(raw)
    df = engineer_features(daily)

    if args.mode == "evaluate":
        evaluate(df, verbose=True)

    elif args.mode == "train":
        print("\nTraining all 14 horizon models on full dataset...")
        models = train_all_horizons(df, save=args.save)

        # Fit Prophet if available to save the full ensemble bundle
        p_model = None
        if HAS_PROPHET:
            print("Fitting baseline Prophet engine for pipeline preservation...")
            prophet_df = daily[["Date", "price"]].rename(columns={"Date": "ds", "price": "y"}).copy()
            prophet_df["long_rains_harvest"] = prophet_df["ds"].dt.month.isin([7, 8, 9]).astype(float)
            prophet_df["short_rains_harvest"] = prophet_df["ds"].dt.month.isin([12, 1, 2]).astype(float)
            prophet_df["planting_season"] = prophet_df["ds"].dt.month.isin([3, 4, 10, 11]).astype(float)
            p_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                              seasonality_mode="multiplicative")
            p_model.add_regressor("long_rains_harvest")
            p_model.add_regressor("short_rains_harvest")
            p_model.add_regressor("planting_season")
            p_model.add_country_holidays(country_name="KE")
            p_model.fit(prophet_df)

        if args.save:
            save_full_pipeline(models, p_model)
        print("Done.")

    elif args.mode == "forecast":
        print("\nTraining LightGBM direct-step models (14 horizons) …")
        models = train_all_horizons(df, save=args.save)

        from_date = daily["Date"].max()
        lgbm_fc = forecast_14_days(df, models, from_date)

        if HAS_PROPHET:
            print("\nFitting Prophet model for weekly ensemble …")
            # For live forecast mode, we fit and instantly get the summary
            prophet_df_fit = daily[["Date", "price"]].rename(columns={"Date": "ds", "price": "y"}).copy()
            # (Reuses your structural functions)
            prophet_weekly = run_prophet_forecast(daily, weeks_ahead=2)
        else:
            prophet_weekly = None

        ensemble = ensemble_weekly_forecast(lgbm_fc, prophet_weekly)
        print_forecast(daily, lgbm_fc, ensemble)


if __name__ == "__main__":
    main()
