"""Shared tabular features and sklearn preprocessor for CT depth experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def ensure_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age_sq and male_age (sex==male times age) from age_years and sex."""
    out = df.copy()
    age = pd.to_numeric(out["age_years"], errors="coerce")
    sex = out["sex"].astype(str).str.lower()
    out["age_sq"] = age**2
    out["male_age"] = (sex == "male").astype(np.float64) * age
    return out


def x_feature_columns(use_engineered: bool) -> list[str]:
    cols = ["sex", "age_years"]
    if use_engineered:
        cols.extend(["age_sq", "male_age"])
    return cols


def build_preprocessor(use_engineered: bool) -> ColumnTransformer:
    num_cols = ["age_years", "age_sq", "male_age"] if use_engineered else ["age_years"]
    cat_cols = ["sex"]
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", ohe, cat_cols),
        ]
    )
