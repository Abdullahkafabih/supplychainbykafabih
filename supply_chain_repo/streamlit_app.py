import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

import streamlit.components.v1 as components
import os
import time



# =============================
# CONFIG
# =============================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "supply_chain_data.csv"

st.set_page_config(page_title="Supply Chain & Logistics BI Dashboard", layout="wide")


# =============================
# STYLE (CSS)
# =============================
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .small-note { opacity: .85; font-size: .9rem; }
    .kpi-card {
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: 14px 16px;
        background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
        box-shadow: 0 6px 20px rgba(0,0,0,.25);
    }
    .kpi-title { font-size: .78rem; letter-spacing: .08em; opacity: .85; }
    .kpi-value { font-size: 1.55rem; font-weight: 800; margin-top: 2px; }
    .kpi-sub { font-size: .85rem; opacity: .8; margin-top: 2px; }
    .badge-ok { display:inline-block; padding: 4px 10px; border-radius: 999px;
                background: rgba(0, 200, 120, .15); border: 1px solid rgba(0, 200, 120, .30); }
    .badge-breach { display:inline-block; padding: 4px 10px; border-radius: 999px;
                    background: rgba(255, 80, 80, .15); border: 1px solid rgba(255, 80, 80, .30); }
    .badge-warn { display:inline-block; padding: 4px 10px; border-radius: 999px;
                  background: rgba(255, 190, 60, .15); border: 1px solid rgba(255, 190, 60, .30); }
    hr { border: none; height: 1px; background: rgba(255,255,255,.08); margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================
# HELPERS
# =============================
def to_snake(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def money(x) -> str:
    try:
        return f"${float(x):,.0f}"
    except Exception:
        return "-"

def pct(x) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "-"

def style_status_df(df_in: pd.DataFrame, status_col: str = "status") -> "pd.io.formats.style.Styler":
    """
    Streamlit st.dataframe() meng-escape HTML, jadi jangan pakai <span> badge.
    Ini cara bener: pakai pandas Styler untuk warnain cell.
    """
    df = df_in.copy()

    def _style_cell(val):
        if val == "OK":
            return "background-color: rgba(0,200,120,.18); border: 1px solid rgba(0,200,120,.25);"
        if val == "WARNING":
            return "background-color: rgba(255,190,60,.18); border: 1px solid rgba(255,190,60,.25);"
        if val == "BREACH":
            return "background-color: rgba(255,80,80,.18); border: 1px solid rgba(255,80,80,.25);"
        if val == "LOW_DATA":
            return "background-color: rgba(180,180,180,.10); border: 1px solid rgba(180,180,180,.18);"
        return ""

    if status_col in df.columns:
        styler = df.style.applymap(_style_cell, subset=[status_col])
    else:
        styler = df.style

    return styler


# =============================
# 1) ETL + 2) DATA CLEANING
# =============================
@st.cache_data(show_spinner=False)
def geocode_locations_to_df(locations: list[str], cache_path: str = "data/geocode_cache.csv") -> pd.DataFrame:
    """
    Mengubah list 'location' (nama kota/negara) menjadi lat/lon via Nominatim (OpenStreetMap).
    Hasil disimpan ke CSV cache supaya tidak geocode ulang.
    """
    # Bersihin dan unik
    locs = [str(x).strip() for x in locations if str(x).strip() and str(x).strip().lower() != "nan"]
    locs = sorted(list(dict.fromkeys(locs)))

    # Load cache kalau ada
    cache = pd.DataFrame(columns=["location", "lat", "lon"])
    if os.path.exists(cache_path):
        try:
            cache = pd.read_csv(cache_path)
            cache["location"] = cache["location"].astype(str).str.strip()
        except Exception:
            cache = pd.DataFrame(columns=["location", "lat", "lon"])

    cached_set = set(cache["location"].astype(str).tolist())
    to_fetch = [l for l in locs if l not in cached_set]

    # Kalau semua sudah ada di cache
    if len(to_fetch) == 0:
        return cache.drop_duplicates("location", keep="last")

    # Geocode (butuh internet)
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="streamlit_supply_chain_dashboard")
    except Exception:
        # geopy belum terinstall
        return cache.drop_duplicates("location", keep="last")

    new_rows = []
    for i, loc in enumerate(to_fetch, start=1):
        try:
            g = geolocator.geocode(loc, timeout=10)
            if g is not None:
                new_rows.append({"location": loc, "lat": float(g.latitude), "lon": float(g.longitude)})
            else:
                new_rows.append({"location": loc, "lat": np.nan, "lon": np.nan})
        except Exception:
            new_rows.append({"location": loc, "lat": np.nan, "lon": np.nan})

        # Rate limiting halus (biar nggak diblok)
        time.sleep(1.0)

    new_df = pd.DataFrame(new_rows)
    out = pd.concat([cache, new_df], ignore_index=True).drop_duplicates("location", keep="last")

    # Simpan cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        out.to_csv(cache_path, index=False)
    except Exception:
        pass

    return out

@st.cache_data
def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    # 1a Extract
    df = pd.read_csv(csv_path)

    # 2d Standardize string headers
    df.columns = [to_snake(c) for c in df.columns]

    # 2d Standardize string values
    cat_cols = [
        "product_type", "customer_demographics", "shipping_carriers",
        "supplier_name", "location", "inspection_results",
        "transportation_modes", "routes"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # 2c Convert numeric + 2b fix flawed data (negative -> NaN)
    num_cols = [
        "price", "availability", "number_of_products_sold", "revenue_generated",
        "stock_levels", "lead_times", "order_quantities", "shipping_times",
        "shipping_costs", "lead_time", "production_volumes", "manufacturing_lead_time",
        "manufacturing_costs", "defect_rates", "costs"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] < 0, c] = np.nan

    # 2b Handle missing numeric with median
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())

    # 1b Transform: feature engineering (BI metrics)
    df["logistics_cost"] = df.get("shipping_costs", 0) + df.get("costs", 0)
    df["total_manufacturing_cost"] = df.get("manufacturing_costs", 0)
    df["total_cost"] = df["logistics_cost"] + df["total_manufacturing_cost"]

    df["profit"] = df.get("revenue_generated", 0) - df["total_cost"]
    df["profit_margin"] = np.where(
        df.get("revenue_generated", 0) > 0,
        df["profit"] / df["revenue_generated"],
        np.nan
    )

    df["stockout_flag"] = (df.get("stock_levels", 0) < df.get("order_quantities", 0)).astype(int)
    df["fill_rate"] = np.where(
        df.get("order_quantities", 0) > 0,
        np.minimum(df.get("stock_levels", 0), df.get("order_quantities", 0)) / df.get("order_quantities", 1),
        np.nan
    )

    df["on_time_flag"] = (df.get("shipping_times", 0) <= df.get("lead_times", 0)).astype(int)
    df["delay_days"] = df.get("shipping_times", 0) - df.get("lead_times", 0)

    # Synthetic timeline for time-series modules (proxy)
    if "sku" in df.columns:
        df = df.sort_values("sku").reset_index(drop=True)
    end = pd.Timestamp.today().normalize()
    df["date"] = pd.date_range(end=end, periods=len(df), freq="D")

    # 1c Load into BI system: returned DataFrame is now ready for Streamlit analytics
    return df


# =============================
# 2a) DATA INSPECTION SUMMARY (ADDED)
# =============================
def data_inspection_report(df: pd.DataFrame) -> dict:
    report = {}
    report["shape"] = df.shape
    report["missing_by_col"] = df.isna().sum().sort_values(ascending=False)
    report["duplicate_rows"] = int(df.duplicated().sum())
    num = df.select_dtypes(include="number")
    report["numeric_summary"] = num.describe().T
    # outlier quick check using IQR (count per col)
    outlier_counts = {}
    for c in num.columns:
        q1 = num[c].quantile(0.25)
        q3 = num[c].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outlier_counts[c] = int(((num[c] < lo) | (num[c] > hi)).sum())
    report["outlier_counts_iqr"] = pd.Series(outlier_counts).sort_values(ascending=False)
    return report


# =============================
# 3) EDA
# =============================
def eda_correlation(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    return num.corr(numeric_only=True)


# =============================
# SUPPLIER SCORECARD + RECOMMENDATIONS
# =============================
def supplier_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("supplier_name", as_index=False).agg(
        shipments=("sku", "count"),
        on_time_rate=("on_time_flag", "mean"),
        avg_defect=("defect_rates", "mean"),
        avg_logistics_cost=("logistics_cost", "mean"),
        avg_lead_time=("lead_time", "mean"),
        stockout_rate=("stockout_flag", "mean"),
        avg_profit_margin=("profit_margin", "mean"),
    )
    g["score"] = (
        (g["on_time_rate"] * 0.45)
        + ((1 - (g["avg_defect"] / (g["avg_defect"].max() + 1e-9))) * 0.25)
        + ((1 - (g["avg_logistics_cost"] / (g["avg_logistics_cost"].max() + 1e-9))) * 0.20)
        + ((1 - g["stockout_rate"]) * 0.10)
    )
    return g.sort_values("score", ascending=False)


def actionable_recommendations(df: pd.DataFrame) -> list[str]:
    recs = []
    sc = supplier_scorecard(df)

    if len(sc) > 0:
        worst_ot = sc.sort_values("on_time_rate").iloc[0]
        if worst_ot["on_time_rate"] < 0.80:
            recs.append(
                f"Prioritas: perbaiki ketepatan waktu supplier '{worst_ot['supplier_name']}' (on-time {worst_ot['on_time_rate']:.0%}). "
                "Aksi: review SLA, jadwal pickup, tambah buffer stock SKU kritikal."
            )

        worst_def = sc.sort_values("avg_defect", ascending=False).iloc[0]
        recs.append(
            f"Quality: supplier '{worst_def['supplier_name']}' defect tinggi ({worst_def['avg_defect']:.2f}). "
            "Aksi: perketat inspeksi inbound + root cause analysis + supplier development."
        )

    if "routes" in df.columns and df["routes"].nunique() > 0:
        top_route = df.groupby("routes")["logistics_cost"].mean().sort_values(ascending=False).index[0]
        recs.append(
            f"Biaya tertinggi di route {top_route}. Aksi: audit rute, konsolidasi shipment (hindari banyak kecil), evaluasi carrier premium."
        )

    if "transportation_modes" in df.columns and df["transportation_modes"].nunique() > 0:
        top_mode = df.groupby("transportation_modes")["logistics_cost"].mean().sort_values(ascending=False).index[0]
        recs.append(
            f"Mode paling mahal: {top_mode}. Aksi: untuk non-urgent shipment, pindah mode lebih murah; buat rule pemilihan mode berdasarkan urgensi."
        )

    sr = float(df["stockout_flag"].mean()) if len(df) else 0.0
    if sr > 0.15:
        recs.append(
            f"Stockout rate {sr:.0%} tinggi. Aksi: set reorder point + safety stock berbasis lead time & demand (SKU high-volume prioritas)."
        )

    if "sku" in df.columns and len(df) > 0:
        low_margin = df.sort_values("profit_margin").iloc[0]
        recs.append(
            f"SKU {low_margin['sku']} margin terendah (≈{low_margin['profit_margin']:.0%}). Aksi: cek pricing, biaya produksi, route/mode shipping."
        )

    return recs[:7]


# =============================
# 4) LINEAR REGRESSION (Drivers of Logistics Cost)
# =============================
def regression_drivers(df: pd.DataFrame):
    y = df["logistics_cost"]
    X = df[
        [
            "product_type", "shipping_carriers", "supplier_name", "location",
            "transportation_modes", "routes",
            "shipping_times", "lead_times", "order_quantities", "stock_levels",
            "defect_rates", "manufacturing_lead_time"
        ]
    ].copy()

    cat_cols = ["product_type", "shipping_carriers", "supplier_name", "location", "transportation_modes", "routes"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
        ]
    )
    model = Pipeline([("prep", pre), ("lr", LinearRegression())])

    # deterministic split
    idx = np.arange(len(df))
    test_mask = (idx % 5 == 0)
    X_train, X_test = X.loc[~test_mask], X.loc[test_mask]
    y_train, y_test = y.loc[~test_mask], y.loc[test_mask]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {"r2": r2_score(y_test, pred), "mae": mean_absolute_error(y_test, pred)}

    ohe = model.named_steps["prep"].named_transformers_["cat"]
    feat_names = np.concatenate([ohe.get_feature_names_out(cat_cols), np.array(num_cols)])
    coef = model.named_steps["lr"].coef_

    coef_df = pd.DataFrame({"feature": feat_names, "coef": coef})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).head(20)

    resid = (y_test.values - pred)
    resid_df = pd.DataFrame({"y_true": y_test.values, "y_pred": pred, "residual": resid})
    return metrics, coef_df, resid_df


# 4d) SIMPLE LINEAR REGRESSION (ADDED)
def simple_lr(df: pd.DataFrame, x_col: str, y_col: str = "logistics_cost"):
    dd = df[[x_col, y_col]].dropna().copy()
    if len(dd) < 5:
        return None

    X = dd[[x_col]].values
    y = dd[y_col].values
    lr = LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    r2 = r2_score(y, pred)

    out = {
        "model": lr,
        "r2": float(r2),
        "coef": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
        "df": dd.assign(pred=pred, residual=y - pred),
    }
    return out


# 4b) REGRESSION ASSUMPTION CHECKS (ADDED)
def assumption_checks(resid_df: pd.DataFrame):
    # normality QQ plot + residual hist + DW and BP test (if available)
    resid = resid_df["residual"].astype(float).values
    y_pred = resid_df["y_pred"].astype(float).values

    results = {}
    # Durbin-Watson
    try:
        from statsmodels.stats.stattools import durbin_watson
        results["durbin_watson"] = float(durbin_watson(resid))
    except Exception:
        results["durbin_watson"] = None

    # Breusch-Pagan
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        exog = sm.add_constant(y_pred)
        bp = het_breuschpagan(resid, exog)
        # bp returns (lm, lm_pvalue, fvalue, f_pvalue)
        results["bp_lm_pvalue"] = float(bp[1])
        results["bp_f_pvalue"] = float(bp[3])
    except Exception:
        results["bp_lm_pvalue"] = None
        results["bp_f_pvalue"] = None

    return results


# =============================
# 5) CLUSTER ANALYSIS
# =============================
def supplier_clustering(sc: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    feat = ["on_time_rate", "avg_defect", "avg_logistics_cost", "avg_lead_time", "stockout_rate"]
    X = sc[feat].copy()
    Xs = (X - X.mean()) / (X.std() + 1e-9)
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    out = sc.copy()
    out["cluster"] = km.fit_predict(Xs)
    return out

# 5c) HIERARCHICAL CLUSTERING (ADDED)
def hierarchical_clustering(sc: pd.DataFrame, k: int = 4):
    feat = ["on_time_rate", "avg_defect", "avg_logistics_cost", "avg_lead_time", "stockout_rate"]
    X = sc[feat].copy()
    Xs = (X - X.mean()) / (X.std() + 1e-9)

    agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = agg.fit_predict(Xs)

    out = sc.copy()
    out["h_cluster"] = labels

    # optional dendrogram if scipy available
    dendro_fig = None
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(Xs, method="ward")
        fig, ax = plt.subplots(figsize=(10, 4))
        dendrogram(Z, ax=ax, no_labels=True, color_threshold=None)
        ax.set_title("Hierarchical Clustering Dendrogram (Ward)")
        ax.set_xlabel("Suppliers")
        ax.set_ylabel("Distance")
        dendro_fig = fig
    except Exception:
        dendro_fig = None

    return out, dendro_fig


# =============================
# 6) TIME SERIES (ARIMA + TS REGRESSION ADDED)
# =============================
def arima_forecast(df: pd.DataFrame, horizon: int = 14):
    daily = df.groupby("date", as_index=False)["logistics_cost"].sum().sort_values("date")
    ts = daily.set_index("date")["logistics_cost"].asfreq("D").fillna(0)

    model = ARIMA(ts, order=(1, 1, 1)).fit()
    fc = model.get_forecast(steps=horizon).summary_frame()

    future_idx = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    future = pd.DataFrame({
        "date": future_idx,
        "forecast": fc["mean"].values,
        "lower": fc["mean_ci_lower"].values,
        "upper": fc["mean_ci_upper"].values,
    })

    hist = daily.rename(columns={"logistics_cost": "actual"})
    hist["forecast"] = np.nan
    hist["lower"] = np.nan
    hist["upper"] = np.nan

    out = pd.concat([hist, future], ignore_index=True)
    return out, ts, model

def simulate_forecast_paths(model, start_value: float, steps: int, n_paths: int = 25, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    paths = []
    for _ in range(n_paths):
        np.random.seed(int(rng.integers(1, 1_000_000)))
        sim = model.simulate(nsimulations=steps)
        sim = np.asarray(sim, dtype=float)
        if len(sim) > 0:
            sim = sim - sim[0] + start_value
        paths.append(sim)

    paths = np.vstack(paths) if len(paths) else np.empty((0, steps))
    cols = [f"t+{i+1}" for i in range(steps)]
    return pd.DataFrame(paths, columns=cols)

# 6a) TIME SERIES LINEAR REGRESSION (ADDED)
def time_series_trend_regression(df: pd.DataFrame):
    daily = df.groupby("date", as_index=False)["logistics_cost"].sum().sort_values("date")
    if len(daily) < 5:
        return None
    daily = daily.copy()
    daily["t"] = np.arange(len(daily))
    X = daily[["t"]].values
    y = daily["logistics_cost"].values
    lr = LinearRegression()
    lr.fit(X, y)
    daily["trend_pred"] = lr.predict(X)
    return {
        "df": daily,
        "coef": float(lr.coef_[0]),
        "intercept": float(lr.intercept_),
        "r2": float(r2_score(y, daily["trend_pred"].values)),
    }


# =============================
# BI DECISION MODULES
# =============================
def z_value(service_level: float) -> float:
    z_map = {0.90: 1.28, 0.95: 1.65, 0.97: 1.88, 0.99: 2.33}
    return z_map.get(service_level, 1.65)

def inventory_policy(df: pd.DataFrame, service_level: float = 0.95) -> pd.DataFrame:
    z = z_value(service_level)

    g = df.groupby("sku", as_index=False).agg(
        avg_demand=("order_quantities", "mean"),
        std_demand=("order_quantities", "std"),
        avg_lead=("lead_times", "mean"),
        stock=("stock_levels", "mean"),
        avg_revenue=("revenue_generated", "mean"),
        avg_sold=("number_of_products_sold", "mean"),
        stockout_rate=("stockout_flag", "mean"),
        on_time=("on_time_flag", "mean"),
    )
    g["std_demand"] = g["std_demand"].fillna(0)
    g["avg_lead"] = g["avg_lead"].clip(lower=0)

    g["safety_stock"] = z * g["std_demand"] * np.sqrt(g["avg_lead"].clip(lower=1))
    g["reorder_point"] = (g["avg_demand"] * g["avg_lead"]) + g["safety_stock"]
    g["recommended_reorder_qty"] = np.maximum(0, g["reorder_point"] - g["stock"]).round(0)

    g["unit_rev"] = np.where(g["avg_sold"] > 0, g["avg_revenue"] / g["avg_sold"], np.nan)
    g["unit_rev"] = g["unit_rev"].fillna(g["unit_rev"].median(skipna=True))

    g["lost_units_proxy"] = np.maximum(g["avg_demand"] - g["stock"], 0)
    g["lost_revenue_proxy"] = g["lost_units_proxy"] * g["unit_rev"]

    g["priority_score"] = (g["stockout_rate"] * 0.6) + (
        g["lost_revenue_proxy"] / (g["lost_revenue_proxy"].max() + 1e-9)
    ) * 0.4

    return g.sort_values("priority_score", ascending=False)

def sla_alerts(df: pd.DataFrame,
              target_on_time: float,
              target_defect: float,
              target_stockout: float,
              min_shipments: int = 5):

    supplier = df.groupby("supplier_name", as_index=False).agg(
        shipments=("sku", "count"),
        on_time=("on_time_flag", "mean"),
        defect=("defect_rates", "mean"),
        stockout=("stockout_flag", "mean"),
        avg_cost=("logistics_cost", "mean"),
    )

    supplier["v_on_time"] = (supplier["on_time"] < target_on_time).astype(int)
    supplier["v_defect"] = (supplier["defect"] > target_defect).astype(int)
    supplier["v_stockout"] = (supplier["stockout"] > target_stockout).astype(int)
    supplier["violations"] = supplier["v_on_time"] + supplier["v_defect"] + supplier["v_stockout"]

    supplier["status"] = np.select(
        [
            supplier["shipments"] < min_shipments,
            supplier["violations"] >= 2,
            supplier["violations"] == 1
        ],
        ["LOW_DATA", "BREACH", "WARNING"],
        default="OK"
    )

    route = df.groupby("routes", as_index=False).agg(
        shipments=("sku", "count"),
        on_time=("on_time_flag", "mean"),
        stockout=("stockout_flag", "mean"),
        avg_cost=("logistics_cost", "mean"),
    )

    route["v_on_time"] = (route["on_time"] < target_on_time).astype(int)
    route["v_stockout"] = (route["stockout"] > target_stockout).astype(int)
    route["violations"] = route["v_on_time"] + route["v_stockout"]

    route["status"] = np.select(
        [
            route["shipments"] < min_shipments,
            route["violations"] >= 2,
            route["violations"] == 1
        ],
        ["LOW_DATA", "BREACH", "WARNING"],
        default="OK"
    )

    mode = df.groupby("transportation_modes", as_index=False).agg(
        shipments=("sku", "count"),
        on_time=("on_time_flag", "mean"),
        avg_cost=("logistics_cost", "mean"),
    )

    order_map = {"BREACH": 0, "WARNING": 1, "OK": 2, "LOW_DATA": 3}
    supplier["_ord"] = supplier["status"].map(order_map)
    route["_ord"] = route["status"].map(order_map)

    supplier = supplier.sort_values(["_ord", "avg_cost"], ascending=[True, False]).drop(columns=["_ord"])
    route = route.sort_values(["_ord", "avg_cost"], ascending=[True, False]).drop(columns=["_ord"])
    mode = mode.sort_values("avg_cost", ascending=False)

    return supplier, route, mode

def what_if_mode_shift(df: pd.DataFrame, from_mode: str, to_mode: str, shift_pct: float):
    if from_mode not in df["transportation_modes"].unique() or to_mode not in df["transportation_modes"].unique():
        return 0.0, None

    avg_from = df.loc[df["transportation_modes"] == from_mode, "logistics_cost"].mean()
    avg_to = df.loc[df["transportation_modes"] == to_mode, "logistics_cost"].mean()
    n_from = int((df["transportation_modes"] == from_mode).sum())

    shift_n = int(round(n_from * shift_pct))
    savings_per = max(0.0, avg_from - avg_to)
    est_savings = shift_n * savings_per

    detail = {
        "from_mode": from_mode,
        "to_mode": to_mode,
        "avg_cost_from": float(avg_from),
        "avg_cost_to": float(avg_to),
        "shipments_from": n_from,
        "shift_shipments": shift_n,
        "savings_per_shipment": float(savings_per),
        "estimated_savings": float(est_savings),
    }
    return float(est_savings), detail


# =============================
# LOAD DATA
# =============================
df = load_and_clean_data(DATA_PATH)


# =============================
# SIDEBAR: FILTERS + TARGETS
# =============================
with st.sidebar:
    st.title("⚙️ Controls")

    st.markdown("**Filters**")
    product_types = sorted(df["product_type"].unique())
    suppliers = sorted(df["supplier_name"].unique())
    modes = sorted(df["transportation_modes"].unique())
    routes = sorted(df["routes"].unique())
    locations = sorted(df["location"].unique())

    f_product = st.multiselect("Product type", product_types, product_types)
    f_supplier = st.multiselect("Supplier", suppliers, suppliers)
    f_mode = st.multiselect("Transportation mode", modes, modes)
    f_route = st.multiselect("Route", routes, routes)
    f_loc = st.multiselect("Location", locations, locations)

    st.markdown("---")
    st.markdown("**Targets (SLA / KPI)**")
    TARGET_ON_TIME = st.slider("Target On-time Rate", 0.70, 0.99, 0.95, 0.01)
    TARGET_STOCKOUT = st.slider("Target Stockout Rate (max)", 0.00, 0.50, 0.05, 0.01)
    TARGET_DEFECT = st.slider("Target Defect Rate (max)", 0.0, 10.0, 2.0, 0.1)

    st.markdown("**Alert sensitivity**")
    MIN_SHIPMENTS = st.slider("Min shipments per entity (avoid LOW_DATA noise)", 1, 30, 5, 1)

    SERVICE_LEVEL = st.selectbox("Service Level (Safety Stock)", [0.90, 0.95, 0.97, 0.99], index=1)

    st.markdown("---")
    st.markdown(
        "<div class='small-note'>BI Mode: dashboard ini menghasilkan <b>alert + action plan + simulasi savings</b>.</div>",
        unsafe_allow_html=True
    )


d = df[
    df["product_type"].isin(f_product)
    & df["supplier_name"].isin(f_supplier)
    & df["transportation_modes"].isin(f_mode)
    & df["routes"].isin(f_route)
    & df["location"].isin(f_loc)
].copy()

if len(d) == 0:
    st.title("📦 Supply Chain & Logistics — Business Intelligence Dashboard")
    st.warning("Data kosong setelah filter. Coba reset filter di sidebar (pilih semua dulu).")
    st.stop()


# =============================
# HEADER
# =============================
st.title("📦 Supply Chain & Logistics — Business Intelligence Dashboard ")
st.caption("Built by Kafabih")
st.caption("Tujuan: monitor inventory, shipping performance, supplier performance, dan transportation cost untuk menemukan bottleneck & meningkatkan efisiensi keputusan.")


# =============================
# KPI CARDS
# =============================
total_revenue = float(d["revenue_generated"].sum())
total_log_cost = float(d["logistics_cost"].sum())
on_time_rate = float(d["on_time_flag"].mean())
stockout_rate = float(d["stockout_flag"].mean())
profit_total = float(d["profit"].sum())

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(f"<div class='kpi-card'><div class='kpi-title'>TOTAL REVENUE</div><div class='kpi-value'>{money(total_revenue)}</div><div class='kpi-sub'>Penjualan (dataset)</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='kpi-card'><div class='kpi-title'>LOGISTICS COST</div><div class='kpi-value'>{money(total_log_cost)}</div><div class='kpi-sub'>Shipping + transport</div></div>", unsafe_allow_html=True)

ot_badge = "badge-ok" if on_time_rate >= TARGET_ON_TIME else "badge-breach"
k3.markdown(f"<div class='kpi-card'><div class='kpi-title'>ON-TIME RATE</div><div class='kpi-value'>{pct(on_time_rate)}</div><div class='kpi-sub'><span class='{ot_badge}'>Target {pct(TARGET_ON_TIME)}</span></div></div>", unsafe_allow_html=True)

so_badge = "badge-ok" if stockout_rate <= TARGET_STOCKOUT else "badge-breach"
k4.markdown(f"<div class='kpi-card'><div class='kpi-title'>STOCKOUT RATE</div><div class='kpi-value'>{pct(stockout_rate)}</div><div class='kpi-sub'><span class='{so_badge}'>Target max {pct(TARGET_STOCKOUT)}</span></div></div>", unsafe_allow_html=True)

k5.markdown(f"<div class='kpi-card'><div class='kpi-title'>TOTAL PROFIT (proxy)</div><div class='kpi-value'>{money(profit_total)}</div><div class='kpi-sub'>Revenue - (log + mfg cost)</div></div>", unsafe_allow_html=True)


# =============================
# TABS (ADDED tab5)
# =============================
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["0) Decision Center", "1) Operations", "2) Supplier", "3) Cost & Bottleneck", "4) Advanced Analytics", "5) BI Documentation & Mapping"]
)


# =============================
# TAB 0: DECISION CENTER
# =============================
with tab0:
    st.subheader("Decision Center — Alerts, Action Plan, dan What-if")

    sup_alert, route_alert, mode_alert = sla_alerts(d, TARGET_ON_TIME, TARGET_DEFECT, TARGET_STOCKOUT, min_shipments=MIN_SHIPMENTS)

    breach_sup = int((sup_alert["status"] == "BREACH").sum())
    warn_sup = int((sup_alert["status"] == "WARNING").sum())
    breach_route = int((route_alert["status"] == "BREACH").sum())
    warn_route = int((route_alert["status"] == "WARNING").sum())

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Supplier BREACH", breach_sup)
    a2.metric("Supplier WARNING", warn_sup)
    a3.metric("Route BREACH", breach_route)
    a4.metric("Service Level", f"{SERVICE_LEVEL:.0%}")

    st.markdown("### A) SLA Alerts (prioritas perbaikan)")
    cL, cR = st.columns(2)

    show_only_issues = st.checkbox("Tampilkan hanya WARNING/BREACH", value=True)

    sup_view = sup_alert.copy()
    route_view = route_alert.copy()

    if show_only_issues:
        sup_view = sup_view[sup_view["status"].isin(["WARNING", "BREACH"])].copy()
        route_view = route_view[route_view["status"].isin(["WARNING", "BREACH"])].copy()

    if "violations" in sup_view.columns and "avg_cost" in sup_view.columns:
        sup_view = sup_view.sort_values(["violations", "avg_cost"], ascending=[False, False])
    if "violations" in route_view.columns and "avg_cost" in route_view.columns:
        route_view = route_view.sort_values(["violations", "avg_cost"], ascending=[False, False])

    with cL:
        st.write("**Supplier status SLA**")
        st.caption("violations = jumlah KPI yang melanggar target (on-time/defect/stockout). Urutan: paling gawat lalu paling mahal.")
        sup_cols = ["supplier_name","shipments","on_time","defect","stockout","violations","avg_cost","status"]
        sup_cols = [c for c in sup_cols if c in sup_view.columns]
        st.dataframe(style_status_df(sup_view[sup_cols].round(3), status_col="status"), use_container_width=True)

    with cR:
        st.write("**Route status SLA**")
        st.caption("violations = jumlah KPI yang melanggar target (on-time/stockout). Urutan: paling gawat lalu paling mahal.")
        route_cols = ["routes","shipments","on_time","stockout","violations","avg_cost","status"]
        route_cols = [c for c in route_cols if c in route_view.columns]
        st.dataframe(style_status_df(route_view[route_cols].round(3), status_col="status"), use_container_width=True)

    st.markdown("### B) Inventory Action Plan (ROP + Safety Stock)")
    pol = inventory_policy(d, service_level=SERVICE_LEVEL)

    st.caption("Rumus: SafetyStock=z·σ(demand)·√L, ROP=μ(demand)·L + SafetyStock, ReorderQty=max(0, ROP - stock).")
    st.dataframe(
        pol.head(20)[
            ["sku", "stock", "avg_demand", "avg_lead", "safety_stock", "reorder_point", "recommended_reorder_qty",
             "stockout_rate", "lost_revenue_proxy", "priority_score"]
        ].round(2),
        use_container_width=True,
    )

    csv = pol.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Inventory Action Plan (CSV)",
        data=csv,
        file_name="inventory_action_plan.csv",
        mime="text/csv",
    )

    st.markdown("### C) What-if: Mode Shift Savings (hemat biaya)")
    m1, m2, m3 = st.columns(3)
    from_mode = m1.selectbox("Shift FROM", sorted(d["transportation_modes"].unique()))
    to_mode_choices = [x for x in sorted(d["transportation_modes"].unique()) if x != from_mode]
    to_mode = m2.selectbox("Shift TO", to_mode_choices if to_mode_choices else sorted(d["transportation_modes"].unique()))
    shift_pct = m3.slider("Shift % shipment", 0.0, 0.50, 0.10, 0.01)

    est_sav, detail = what_if_mode_shift(d, from_mode=from_mode, to_mode=to_mode, shift_pct=shift_pct)
    if detail is not None:
        st.success(
            f"Estimasi savings: {money(detail['estimated_savings'])} "
            f"(pindahkan {detail['shift_shipments']} shipment dari {detail['from_mode']} → {detail['to_mode']})."
        )
        st.json({k: (round(v, 2) if isinstance(v, float) else v) for k, v in detail.items()})
    else:
        st.info("Mode pilihan tidak tersedia pada data yang sedang difilter.")

    st.markdown("### D) Executive Summary (putusan cepat)")
    st.write(
        f"- Stockout Rate: **{pct(stockout_rate)}** (target max {pct(TARGET_STOCKOUT)}) "
        + ("✅ OK" if stockout_rate <= TARGET_STOCKOUT else "🚨 BREACH — fokus replenishment & safety stock")
    )
    st.write(
        f"- On-time Rate: **{pct(on_time_rate)}** (target {pct(TARGET_ON_TIME)}) "
        + ("✅ OK" if on_time_rate >= TARGET_ON_TIME else "🚨 BREACH — review SLA + pickup schedule + buffer")
    )
    st.write(
        f"- Supplier issues: **{warn_sup + breach_sup}** (WARN {warn_sup} | BREACH {breach_sup}) "
        f"| Route issues: **{warn_route + breach_route}** (WARN {warn_route} | BREACH {breach_route})"
    )

    st.markdown("**Urutan prioritas aksi (rekomendasi BI):**")
    st.write("1) Jalankan Inventory Action Plan untuk SKU prioritas (turunkan stockout & lost revenue).")
    st.write("2) Tangani Supplier/Route dengan violations tertinggi dan avg_cost tertinggi (negosiasi SLA, QC, pickup schedule).")
    st.write("3) Jalankan optimasi biaya lewat mode shift untuk shipment non-urgent (hemat biaya).")


# =============================
# TAB 1: OPERATIONS
# =============================
with tab1:
    st.subheader("Inventory Risk: Top 20 SKU Stock vs Order (indikasi stockout)")
    inv = d.copy()
    inv["gap"] = inv["stock_levels"] - inv["order_quantities"]
    inv = inv.sort_values("gap").head(20)

    fig_inv = px.bar(inv, x="sku", y=["stock_levels", "order_quantities"], barmode="group")
    if len(fig_inv.data) >= 2:
        fig_inv.data[0].name = "stock_levels"
        fig_inv.data[0].marker.color = "#4EA3FF"  # biru
        fig_inv.data[1].name = "order_quantities"
        fig_inv.data[1].marker.color = "#FF5A5A"  # merah
    fig_inv.update_layout(legend_title_text="variable")
    st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown(
        "**Interpretasi:** jika bar merah (order) > biru (stock), ada risiko stockout/backorder. "
        "Fokus replenishment untuk SKU dengan gap paling negatif."
    )

    st.subheader("On-time Performance: Shipping time vs Promised lead time")
    fig_ot = px.scatter(
        d,
        x="lead_times",
        y="shipping_times",
        color="on_time_flag",
        hover_data=["sku", "supplier_name", "shipping_carriers", "routes", "transportation_modes"],
        labels={"lead_times": "Promised lead time", "shipping_times": "Actual shipping time"},
    )
    x_min, x_max = float(d["lead_times"].min()), float(d["lead_times"].max())
    fig_ot.add_shape(type="line", x0=x_min, y0=x_min, x1=x_max, y1=x_max)
    st.plotly_chart(fig_ot, use_container_width=True)

    st.markdown(
        "**Interpretasi:** titik di bawah garis diagonal = on-time. "
        "Jika banyak titik telat pada lead time kecil, berarti SLA terlalu agresif atau ada bottleneck operasional."
    )


# =============================
# TAB 2: SUPPLIER
# =============================
with tab2:
    st.subheader("Supplier Scorecard (gabungan on-time, defect, cost, stockout)")
    sc = supplier_scorecard(d)

    left, right = st.columns(2)
    with left:
        fig_top = px.bar(sc.head(10), x="supplier_name", y="score", title="Top Supplier Overall Score (weighted)")
        st.plotly_chart(fig_top, use_container_width=True)

    with right:
        fig_bottom = px.bar(
            sc.sort_values("on_time_rate").head(10),
            x="supplier_name",
            y="on_time_rate",
            title="Bottom Supplier On-time Rate (prioritas improvement)",
        )
        st.plotly_chart(fig_bottom, use_container_width=True)

    st.subheader("Supplier yang perlu ditangani dulu (prioritas)")
    st.dataframe(sc.sort_values("score").head(10).round(3), use_container_width=True)

    st.subheader("Actionable Recommendations")
    for r in actionable_recommendations(d):
        st.write("•", r)


# =============================
# TAB 3: COST & BOTTLENECK
# =============================
with tab3:
    st.subheader("Logistics Cost by Transportation Mode")
    mode_cost = (
        d.groupby("transportation_modes", as_index=False)["logistics_cost"]
        .sum()
        .sort_values("logistics_cost", ascending=False)
    )
    st.plotly_chart(px.bar(mode_cost, x="transportation_modes", y="logistics_cost"), use_container_width=True)

    st.subheader("Logistics Cost by Route")
    route_cost = d.groupby("routes", as_index=False)["logistics_cost"].sum().sort_values("logistics_cost", ascending=False)
    st.plotly_chart(px.bar(route_cost, x="routes", y="logistics_cost"), use_container_width=True)

    st.subheader("Top Bottlenecks (cost tinggi + delay + stockout)")
    b = d.copy()
    delay_pos = np.maximum(b["delay_days"], 0)

    b["bottleneck_score"] = (
        (b["logistics_cost"] / (b["logistics_cost"].max() + 1e-9)) * 0.45
        + (delay_pos / (delay_pos.max() + 1e-9)) * 0.35
        + b["stockout_flag"] * 0.20
    )

    cols = [
        "sku", "supplier_name", "shipping_carriers", "transportation_modes", "routes", "location",
        "logistics_cost", "lead_times", "shipping_times", "delay_days",
        "stock_levels", "order_quantities", "bottleneck_score"
    ]
    st.dataframe(b.sort_values("bottleneck_score", ascending=False).head(25)[cols].round(3), use_container_width=True)

    st.markdown(
        "**Cara pakai tabel bottleneck:** ambil baris skor tertinggi → itulah titik paling “mahal + telat + stockout”. "
        "Biasanya ini kandidat pertama untuk ubah rute/mode/carrier atau negosiasi dengan supplier."
    )


# =============================
# TAB 4: ADVANCED ANALYTICS
# =============================
with tab4:
    st.subheader("EDA: Correlation (numeric)")
    corr = eda_correlation(d)
    fig_corr = px.imshow(corr, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

    # 3b/3c/3d tambahan EDA (tanpa menghapus yang sudah ada)
    st.markdown("### EDA Tambahan (sesuai rubrik)")
    e1, e2 = st.columns(2)

    with e1:
        st.write("**Univariate (single variable): distribusi logistics_cost**")
        fig_u = px.histogram(d, x="logistics_cost", nbins=30, title="Distribution of Logistics Cost")
        st.plotly_chart(fig_u, use_container_width=True)

    with e2:
        st.write("**Univariate: boxplot defect_rates**")
        if "defect_rates" in d.columns:
            fig_box = px.box(d, y="defect_rates", title="Defect Rates (Boxplot)")
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Kolom defect_rates tidak ada.")

    st.write("**Bivariate (2 variabel): logistics_cost vs shipping_times**")
    if "shipping_times" in d.columns:
        fig_bi = px.scatter(d, x="shipping_times", y="logistics_cost",
                            hover_data=["sku", "supplier_name", "routes", "transportation_modes"],
                            title="Logistics Cost vs Shipping Time")
        st.plotly_chart(fig_bi, use_container_width=True)

    st.write("**Multivariate: scatter matrix (beberapa variabel sekaligus)**")
    multi_cols = [c for c in ["logistics_cost", "shipping_times", "lead_times", "order_quantities", "stock_levels", "defect_rates"] if c in d.columns]
    if len(multi_cols) >= 3:
        fig_sm = px.scatter_matrix(d, dimensions=multi_cols[:5], color="on_time_flag",
                                   title="Scatter Matrix (Multi-variable EDA)")
        st.plotly_chart(fig_sm, use_container_width=True)
    else:
        st.info("Variabel numeric tidak cukup untuk scatter-matrix.")


    st.subheader("Regression: Drivers of Logistics Cost")
    metrics, coef_df, resid_df = regression_drivers(d)
    st.write(f"R² = {metrics['r2']:.2f} | MAE = {metrics['mae']:.2f}")

    fig_coef = px.bar(coef_df.sort_values("abs_coef"), x="coef", y="feature", orientation="h")
    st.plotly_chart(fig_coef, use_container_width=True)

    fig_res = px.scatter(resid_df, x="y_pred", y="residual", title="Residual Plot (cek pola error)")
    st.plotly_chart(fig_res, use_container_width=True)

    # 4b assumption checks (tambahan)
    st.markdown("### Regression Assumptions (tambahan)")
    checks = assumption_checks(resid_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Durbin-Watson", "-" if checks["durbin_watson"] is None else f"{checks['durbin_watson']:.2f}")
    c2.metric("Breusch-Pagan p (LM)", "-" if checks["bp_lm_pvalue"] is None else f"{checks['bp_lm_pvalue']:.4f}")
    c3.metric("Breusch-Pagan p (F)", "-" if checks["bp_f_pvalue"] is None else f"{checks['bp_f_pvalue']:.4f}")

    # QQ plot + histogram residual (matplotlib) -> memenuhi "python graph"
    fig_q, ax = plt.subplots(figsize=(6, 4))
    sm.qqplot(resid_df["residual"].astype(float).values, line="45", ax=ax)
    ax.set_title("QQ Plot Residuals (Normality check)")
    st.pyplot(fig_q)

    fig_h, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(resid_df["residual"].astype(float).values, bins=30)
    ax2.set_title("Residual Histogram")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Count")
    st.pyplot(fig_h)

    # 4d simple LR (tambahan)
    st.markdown("### Simple Linear Regression (1 variabel)")
    x_choice = st.selectbox("Pilih variabel X untuk simple LR (Y=logistics_cost)",
                            [c for c in ["shipping_times", "lead_times", "order_quantities", "stock_levels", "defect_rates"] if c in d.columns],
                            index=0)
    slr = simple_lr(d, x_col=x_choice, y_col="logistics_cost")
    if slr is None:
        st.info("Data tidak cukup untuk simple regression.")
    else:
        st.write(f"Model: logistics_cost = {slr['intercept']:.2f} + ({slr['coef']:.2f}) * {x_choice}")
        st.write(f"R² (simple LR) = {slr['r2']:.3f}")
        fig_slr = px.scatter(slr["df"], x=x_choice, y="logistics_cost", trendline="ols",
                             title=f"Simple LR: logistics_cost vs {x_choice}")
        st.plotly_chart(fig_slr, use_container_width=True)

    st.subheader("Supplier Clustering (segmentation)")
    sc2 = supplier_clustering(supplier_scorecard(d), k=4)
    fig_cluster = px.scatter(
        sc2,
        x="avg_logistics_cost",
        y="on_time_rate",
        color="cluster",
        size="shipments",
        hover_data=["supplier_name", "avg_defect", "stockout_rate", "avg_lead_time", "score"],
        labels={"avg_logistics_cost": "Avg Logistics Cost", "on_time_rate": "On-time Rate"},
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    # 5c hierarchical clustering (tambahan)
    st.markdown("### Hierarchical Clustering (tambahan)")
    st.caption("Tujuan: memenuhi rubrik 5c. Hierarchical berguna untuk melihat struktur kedekatan supplier tanpa harus memilih K sejak awal.")
    sc_base = supplier_scorecard(d)
    hc_k = st.slider("Jumlah cluster (hierarchical)", 2, 8, 4, 1)
    hc_out, dendro_fig = hierarchical_clustering(sc_base, k=hc_k)
    st.dataframe(hc_out.sort_values(["h_cluster", "score"], ascending=[True, False]).round(3), use_container_width=True)
    if dendro_fig is not None:
        st.pyplot(dendro_fig)
    else:
        st.info("Dendrogram butuh SciPy. Cluster label hierarchical tetap sudah dibuat tanpa dendrogram.")

    st.subheader("ARIMA Forecast (timeline sintetis)")

    cc1, cc2, cc3 = st.columns([1.2, 1.2, 1.0])
    horizon = cc1.slider("Forecast horizon (hari)", 7, 60, 14, 1)
    smooth_win = cc2.slider("Smoothing window (rolling mean)", 1, 14, 6, 1)
    show_paths = cc3.checkbox("Show scenario paths", value=True)

    fc, ts, model = arima_forecast(d, horizon=horizon)

    actual_series = fc["actual"].copy()
    actual_sm = actual_series.rolling(window=smooth_win, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["actual"], mode="lines", name="Actual", opacity=0.55))
    fig.add_trace(go.Scatter(x=fc["date"], y=actual_sm, mode="lines", name=f"Actual (smoothed {smooth_win})"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], mode="lines", name="Forecast"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper"], mode="lines", name="Upper", line=dict(width=0), showlegend=True))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower"], mode="lines", name="Lower", fill="tonexty", line=dict(width=0), showlegend=True))

    if show_paths:
        last_actual = float(ts.iloc[-1]) if len(ts) else 0.0
        paths = simulate_forecast_paths(model, start_value=last_actual, steps=horizon, n_paths=25, seed=42)
        future_dates = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
        for i in range(len(paths)):
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=paths.iloc[i].values,
                    mode="lines",
                    name="Scenario" if i == 0 else None,
                    opacity=0.18,
                    showlegend=(i == 0),
                )
            )

    fig.update_layout(
        title="ARIMA Forecast (timeline sintetis)",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="v"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 6a time series regression (tambahan)
    st.markdown("### Time Series with Linear Regression (tambahan)")
    st.caption("Rubrik 6a: contoh sederhana memodelkan tren (naik/turun) logistics_cost dari waktu ke waktu menggunakan linear regression.")
    ts_lr = time_series_trend_regression(d)
    if ts_lr is None:
        st.info("Data time series harian tidak cukup untuk trend regression.")
    else:
        st.write(f"Trend model: logistics_cost = {ts_lr['intercept']:.2f} + ({ts_lr['coef']:.2f}) * t")
        st.write(f"R² (trend regression) = {ts_lr['r2']:.3f}")
        fig_ts_lr = go.Figure()
        fig_ts_lr.add_trace(go.Scatter(x=ts_lr["df"]["date"], y=ts_lr["df"]["logistics_cost"], mode="lines", name="Actual"))
        fig_ts_lr.add_trace(go.Scatter(x=ts_lr["df"]["date"], y=ts_lr["df"]["trend_pred"], mode="lines", name="Trend (LR)"))
        fig_ts_lr.update_layout(title="Daily Logistics Cost: Actual vs Trend (Linear Regression)")
        st.plotly_chart(fig_ts_lr, use_container_width=True)

    st.info(
        "Catatan: dataset tidak punya tanggal transaksi asli, jadi ARIMA memakai timeline sintetis (urut record). "
        "Mean forecast ARIMA cenderung halus; untuk variasi gunakan scenario paths (simulasi) atau time-series asli."
    )


# =============================
# TAB 5: BI DOCUMENTATION & MAPPING (STREAMLIT ONLY)
# =============================
with tab5:
    st.subheader("BI Documentation (ETL → Cleaning → EDA → Modeling)")

    st.markdown(
        """
        **Ringkas rubrik yang dipenuhi di dashboard ini:**
        - **ETL:** baca CSV → transform fitur → load ke aplikasi BI (Streamlit).
        - **Cleaning:** standardisasi string, casting numerik, tangani nilai negatif & missing.
        - **EDA:** univariate, bivariate, multivariate, correlation.
        - **Regression:** model driver biaya + pengecekan asumsi + simple LR.
        - **Clustering:** k-means + hierarchical (Agglomerative).
        - **Time series:** ARIMA + trend regression.
        - **Visualization:** Plotly interaktif + matplotlib statik.
        - **Geo mapping:** Folium (Leaflet) ditampilkan di Streamlit.
        
        **Catatan:** Dashboard web dikembangkan dengan **Streamlit** (bukan Dash/Shiny).
        """
    )

    st.markdown("### 2a) Data Inspection Summary (tambahan)")
    rep = data_inspection_report(d)
    st.write(f"Shape: **{rep['shape'][0]} rows × {rep['shape'][1]} cols**")
    st.write(f"Duplicate rows: **{rep['duplicate_rows']}**")

    st.write("Missing values per column (top):")
    st.dataframe(rep["missing_by_col"].head(15), use_container_width=True)

    st.write("Numeric summary:")
    st.dataframe(rep["numeric_summary"].round(3), use_container_width=True)

    st.write("Outlier count (IQR rule):")
    st.dataframe(rep["outlier_counts_iqr"].head(15), use_container_width=True)

    st.markdown("### 7c) Geo-mapping (sesuai dataset) — Plotly Geo")
st.caption("Pakai koordinat asli bila ada. Jika tidak ada, sistem akan geocode 'location' → lat/lon (dicache).")

# agregasi per location
loc_agg = (
    d.groupby("location", as_index=False)
    .agg(
        shipments=("sku", "count"),
        avg_logistics_cost=("logistics_cost", "mean"),
        total_logistics_cost=("logistics_cost", "sum"),
        on_time_rate=("on_time_flag", "mean"),
        stockout_rate=("stockout_flag", "mean"),
    )
    .sort_values("total_logistics_cost", ascending=False)
)

# CASE A: dataset punya lat/lon langsung
if ("latitude" in d.columns) and ("longitude" in d.columns):
    pts = (
        d[["latitude", "longitude", "location", "logistics_cost", "on_time_flag", "stockout_flag"]]
        .dropna(subset=["latitude", "longitude"])
        .copy()
    )
    pts["latitude"] = pd.to_numeric(pts["latitude"], errors="coerce")
    pts["longitude"] = pd.to_numeric(pts["longitude"], errors="coerce")
    pts = pts.dropna(subset=["latitude", "longitude"])

    fig_map = px.scatter_geo(
        pts,
        lat="latitude",
        lon="longitude",
        color="logistics_cost",
        hover_name="location",
        hover_data=["on_time_flag", "stockout_flag"],
        title="Shipment Points (raw) — berdasarkan latitude/longitude dataset",
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# CASE B: tidak ada lat/lon → geocode dari 'location'
else:
    # Batasi geocode kalau lokasi terlalu banyak (biar nggak lama & rate limit)
    max_geo = 40
    loc_for_geo = loc_agg.head(max_geo)["location"].astype(str).tolist()

    geo_df = geocode_locations_to_df(loc_for_geo, cache_path="data/geocode_cache.csv")
    loc_plot = loc_agg.merge(geo_df, on="location", how="left")

    ok = loc_plot.dropna(subset=["lat", "lon"]).copy()
    miss = int(loc_plot["lat"].isna().sum())

    if len(ok) == 0:
        st.warning(
            "Tidak ada lokasi yang berhasil di-geocode. "
            "Solusi paling akurat: tambahkan kolom latitude/longitude di dataset."
        )
    else:
        if miss > 0:
            st.info(f"{miss} location belum dapat koordinat (geocode gagal/ambigu). Map menampilkan yang berhasil saja.")

        fig_map = px.scatter_geo(
            ok,
            lat="lat",
            lon="lon",
            size="shipments",
            color="avg_logistics_cost",
            hover_name="location",
            hover_data={
                "shipments": True,
                "avg_logistics_cost": ":.2f",
                "total_logistics_cost": ":.2f",
                "on_time_rate": ":.2f",
                "stockout_rate": ":.2f",
                "lat": False,
                "lon": False,
            },
            title="Geo Map by Location — hasil geocoding 'location' (cached)",
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)

    st.caption("Cache koordinat tersimpan di: data/geocode_cache.csv (biar run berikutnya cepat).")


st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© Supply Chain BI Dashboard — Streamlit | Fokus: keputusan (alert, action plan, savings). Built by Kafabih")

