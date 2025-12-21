from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder


# =========================================================
# Custom transformers (keep these so joblib can unpickle)
# =========================================================

class ModelTargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target-encode `model` into `model_te` using TRAIN y only, then drop `model`.
    """
    def __init__(self, model_col="model", out_col="model_te", drop_model=True):
        self.model_col = model_col
        self.out_col = out_col
        self.drop_model = drop_model

    def fit(self, X, y):
        if y is None:
            raise ValueError("ModelTargetEncoder requires y in fit().")
        X = X.copy()
        s = pd.Series(y, index=X.index)

        mean_by_model = s.groupby(X[self.model_col]).mean()
        self.mean_by_model_ = mean_by_model
        self.global_mean_ = float(s.mean())
        return self

    def transform(self, X):
        X = X.copy()
        X[self.out_col] = X[self.model_col].map(self.mean_by_model_).astype(float)
        X[self.out_col] = X[self.out_col].fillna(self.global_mean_)
        if self.drop_model and self.model_col in X.columns:
            X = X.drop(columns=[self.model_col])
        return X


class TaxImputeBinOHE(BaseEstimator, TransformerMixin):
    """
    1) impute tax by median tax per model_te (learned on TRAIN), fallback global median
    2) bin tax with fixed bins/labels
    3) OHE tax_bin, then drop raw tax and add tax_bin_* dummies
    """
    def __init__(
        self,
        tax_col="tax",
        key_col="model_te",
        bins=(-0.1, 0.1, 100, 170, 450, np.inf),
        labels=("0", "1_100", "101_170", "171_450", "451_plus"),
    ):
        self.tax_col = tax_col
        self.key_col = key_col
        self.bins = list(bins)
        self.labels = list(labels)

    def fit(self, X, y=None):
        X = X.copy()
        tax = pd.to_numeric(X[self.tax_col], errors="coerce")
        key = pd.to_numeric(X[self.key_col], errors="coerce")
        self.tax_median_by_key_ = tax.groupby(key).median()
        self.global_tax_median_ = float(tax.median())

        tax_bin = pd.cut(tax.fillna(self.global_tax_median_), bins=self.bins, labels=self.labels)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.ohe_.fit(tax_bin.to_frame(name="tax_bin"))
        self.tax_ohe_cols_ = list(self.ohe_.get_feature_names_out(["tax_bin"]))
        return self

    def transform(self, X):
        X = X.copy()
        tax = pd.to_numeric(X[self.tax_col], errors="coerce")

        key = pd.to_numeric(X[self.key_col], errors="coerce")
        tax = tax.fillna(key.map(self.tax_median_by_key_))
        tax = tax.fillna(self.global_tax_median_)

        tax_bin = pd.cut(tax, bins=self.bins, labels=self.labels)

        tax_ohe = self.ohe_.transform(tax_bin.to_frame(name="tax_bin"))
        tax_ohe_df = pd.DataFrame(tax_ohe, columns=self.tax_ohe_cols_, index=X.index)

        X = X.drop(columns=[c for c in [self.tax_col] if c in X.columns])
        X = pd.concat([X, tax_ohe_df], axis=1)
        return X


class DataFrameKNNImputer(BaseEstimator, TransformerMixin):
    """KNNImputer that returns a DataFrame with the same columns/index."""
    def __init__(self, n_neighbors=8, weights="uniform"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights=self.weights)
        self.imputer_.fit(X)
        return self

    def transform(self, X):
        arr = self.imputer_.transform(X)
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects a fixed list of columns by name.
    If some selected columns are missing at transform-time, it creates them filled with 0.0.
    """
    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, X, y=None):
        self.cols_ = list(self.cols)
        return self

    def transform(self, X):
        X = X.copy()
        missing = [c for c in self.cols_ if c not in X.columns]
        for c in missing:
            X[c] = 0.0
        return X.loc[:, self.cols_]


# =========================================================
# Configuration + helpers
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_pipeline.joblib"  # adjust if needed

CURRENT_YEAR = 2025

DISPLAY_LABELS = {
    "mileage": "Mileage (Miles)",
    "tax": "Tax (£)",
    "engineSize": "Engine Size (L)",
    "carAge": "Year (Until {CURRENT_YEAR})",
    "mpg": "Miles per Gallon",
}

HIDE_INPUT_COLS = {"previousOwners", "paintQuality%", "paintQuality", "hasDamage", "carID", "carId"}

NUMERIC_NAME_HINTS = {
    "mileage", "mpg", "tax", "enginesize", "engine_size", "carage",
    "previousowners", "year"
}


@st.cache_resource
def load_model_asset(path: Path):
    obj = joblib.load(path)

    # Case A: dict with pipeline + metadata
    if isinstance(obj, dict) and "pipeline" in obj:
        pipe = obj["pipeline"]
        feature_names = obj.get("feature_names", None)
        train_uniques = obj.get("train_uniques", {}) or {}
        schema = obj.get("schema", {}) or {}  # optional: {"col": "numeric"|"categorical"}
        return pipe, feature_names, train_uniques, schema

    # Case B: pipeline saved directly
    return obj, None, {}, {}


def onehot_group(feature_names: list[str], prefix: str):
    cols = [c for c in feature_names if c.startswith(prefix)]
    labels = [c[len(prefix):] for c in cols]
    return cols, labels


def set_onehot_choice(inputs: dict, prefix: str, cols: list[str], choice: str):
    for c in cols:
        inputs[c] = 0.0
    key = f"{prefix}{choice}"
    if key in cols:
        inputs[key] = 1.0


def looks_numeric(col: str, schema: dict) -> bool:
    if schema.get(col) == "numeric":
        return True
    key = col.replace(" ", "").replace("_", "").lower()
    return any(h in key for h in NUMERIC_NAME_HINTS)


def coerce_types(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if looks_numeric(c, schema=schema):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# =========================================================
# UI
# =========================================================

st.title("Cars 4 You")
st.write("Provide the car features below to predict its price.")

try:
    pipe, feature_names, train_uniques, schema = load_model_asset(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

if not feature_names:
    st.error(
        "Your saved joblib does not include `feature_names`, so the app cannot build the input form.\n\n"
        "Fix: save as dict: joblib.dump({'pipeline': pipe, 'feature_names': list(X_train.columns), "
        "'train_uniques': {...}}, ...)"
    )
    st.stop()

# Optional: try pandas outputs if supported (safe)
try:
    pipe.set_output(transform="pandas")
except Exception:
    pass

st.subheader("Inputs")

with st.form("predict_form"):
    # Initialise everything to 0 so we never miss a feature key
    inputs: dict[str, object] = {c: 0 for c in feature_names}

    # --- Collapse Brand / transmission / fuelType into dropdowns (one-hot) ---
    brand_cols, brand_vals = onehot_group(feature_names, "Brand_")
    trans_cols, trans_vals = onehot_group(feature_names, "transmission_")
    fuel_cols, fuel_vals = onehot_group(feature_names, "fuelType_")

    if brand_cols:
        chosen_brand = st.selectbox("Brand", options=sorted(brand_vals), index=0)
        set_onehot_choice(inputs, "Brand_", brand_cols, chosen_brand)

    if trans_cols:
        chosen_trans = st.selectbox("Transmission", options=sorted(trans_vals), index=0)
        set_onehot_choice(inputs, "transmission_", trans_cols, chosen_trans)

    if fuel_cols:
        chosen_fuel = st.selectbox("Fuel type", options=sorted(fuel_vals), index=0)
        set_onehot_choice(inputs, "fuelType_", fuel_cols, chosen_fuel)

    skip_cols = set(brand_cols + trans_cols + fuel_cols)

    # Hide fields from UI (kept fixed at 0.0)
    for c in HIDE_INPUT_COLS:
        if c in feature_names:
            inputs[c] = 0.0
            skip_cols.add(c)

    # Year input -> compute carAge (2025 - year); do not show carAge directly
    if "carAge" in feature_names:
        year_val = st.number_input(
            f"Year (Until {CURRENT_YEAR})",
            min_value=1990,
            max_value=CURRENT_YEAR - 1,
            value=CURRENT_YEAR - 1,
            step=1,
        )
        car_age = float(CURRENT_YEAR - int(year_val))
        inputs["carAge"] = car_age
        skip_cols.add("carAge")
    # ---------------------------------------------------------
    # Model: SINGLE control (selectbox is searchable by typing)
    # (keep outside form if you want it to immediately react)
    # -------------------------------------------------------
    selected_model = ""
    if "model" in feature_names:
       model_opts = list(train_uniques.get("model", []) or [])
       if model_opts:
            selected_model = st.selectbox(
               "Model",
               options=sorted(model_opts),
               index=0,
               key="model_choice",
            )
            st.caption("Tip: click the dropdown and start typing to search.")
       else:
            selected_model = st.text_input("Model", value="", key="model_manual")
        # Model comes from outside-form widgets (live filtering)
    if "model" in feature_names:
        inputs["model"] = st.session_state.get("model_choice", selected_model) or selected_model
        skip_cols.add("model")

    # Remaining features
    for col in feature_names:
        if col in skip_cols:
            continue

        label = DISPLAY_LABELS.get(col, col)

        if looks_numeric(col, schema=schema):
            inputs[col] = st.number_input(label, value=float(inputs.get(col, 0.0)))
        else:
            inputs[col] = st.text_input(label, value=str(inputs.get(col, "")))

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a 1-row dataframe in the exact expected order
    x_new = pd.DataFrame([[inputs[c] for c in feature_names]], columns=feature_names)
    x_new = coerce_types(x_new, schema=schema)

    try:
        pred = pipe.predict(x_new)[0]
        st.subheader("Prediction")
        st.metric("Your car price is approximately", f"£ {pred:,.0f}")
    except Exception as e:
        st.error("Prediction failed. The most common causes are schema mismatch or unexpected input types.")
        st.code(str(e))
        st.write("Debug info (input row):")
        st.dataframe(x_new)


