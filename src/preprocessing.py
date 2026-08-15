"""
preprocessing.py
----------------
Cleaning, feature engineering and the scikit-learn preprocessing pipeline.

Three things happen in this file:
1. clean_data()          -> fixes missing values and clearly invalid records
2. add_time_features()   -> builds Year / Month / Hour / DayOfWeek from Date
3. build_preprocessor()  -> ColumnTransformer used inside the ML pipelines
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Chicago city limits, used only to flag impossible coordinates
CHICAGO_LAT_RANGE = (41.6, 42.1)
CHICAGO_LON_RANGE = (-87.95, -87.5)

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Columns that must NEVER be given to the model.
# Each one is listed together with the reason, which is also shown in the app.
# ---------------------------------------------------------------------------
EXCLUDED_COLUMNS = {
    "Arrest": "This is the target variable itself.",
    "ID": "Unique record identifier - carries no information about arrests.",
    "Case Number": "Unique identifier assigned to the case, not a real feature.",
    "Date": "Raw timestamp; we use the extracted Year/Month/Hour/DayOfWeek instead.",
    "Updated On": "Record was updated AFTER the case was processed - post-arrest information (leakage).",
    "Block": "Free-text address with ~24,000 unique values; District/Beat cover location better.",
    "Location": "Duplicate of the Latitude/Longitude pair stored as text.",
    "X Coordinate": "Duplicate of Longitude in a different projection.",
    "Y Coordinate": "Duplicate of Latitude in a different projection.",
    "Latitude": "Exact coordinates are dropped; District/Beat/Ward already describe the area.",
    "Longitude": "Exact coordinates are dropped; District/Beat/Ward already describe the area.",
    "IUCR": "Offence code that is a direct duplicate of Primary Type + Description.",
    "FBI Code": "Another duplicate coding of the same offence type.",
    "_year": "Exact copy of the existing Year column.",
}


# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------
def clean_data(df):
    """
    Clean the raw dataframe.

    The strategy is deliberately conservative - we do not delete large parts of
    the data. Every decision is recorded in the returned `log` so it can be
    displayed in the Streamlit app and explained during the viva.
    """
    df = df.copy()
    log = []
    rows_before = len(df)

    # --- 1. Exact duplicate rows -------------------------------------------
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        df = df.drop_duplicates()
        log.append(f"Removed {n_dupes:,} exactly duplicated rows.")
    else:
        log.append("No exactly duplicated rows were found.")

    # --- 2. Columns with extremely high missingness -------------------------
    # A column that is almost always empty cannot help the analysis.
    missing_share = df.isna().mean()
    too_empty = [c for c in df.columns if missing_share[c] > 0.60]
    if too_empty:
        df = df.drop(columns=too_empty)
        log.append(f"Dropped columns with more than 60% missing values: {too_empty}.")
    else:
        log.append("No column had more than 60% missing values, so none was dropped.")

    # --- 3. Empty strings are really missing values -------------------------
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            blanks = df[col].astype(str).str.strip().eq("")
            if blanks.any():
                df.loc[blanks, col] = np.nan
                log.append(f"Converted {int(blanks.sum()):,} empty strings in '{col}' to missing values.")

    # --- 4. Impossible coordinates ------------------------------------------
    # Latitude/Longitude of 0 (or outside Chicago) are recording errors.
    if "Latitude" in df.columns and "Longitude" in df.columns:
        bad_coords = (
            df["Latitude"].notna()
            & (
                ~df["Latitude"].between(*CHICAGO_LAT_RANGE)
                | ~df["Longitude"].between(*CHICAGO_LON_RANGE)
            )
        )
        if bad_coords.any():
            # We only blank the coordinates - the rest of the record is still useful.
            df.loc[bad_coords, ["Latitude", "Longitude"]] = np.nan
            log.append(
                f"Blanked {int(bad_coords.sum()):,} coordinates that fall outside the Chicago city limits."
            )
        else:
            log.append("All available coordinates fall inside the Chicago city limits.")

    # --- 5. Fill remaining missing values ------------------------------------
    # Categorical -> "UNKNOWN" (keeps the row and is honest about what we know)
    # Numerical   -> median (robust to outliers, unlike the mean)
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        n_missing = int(df[col].isna().sum())
        if pd.api.types.is_numeric_dtype(df[col]):
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            log.append(f"Filled {n_missing:,} missing values in '{col}' with the median ({median_value:g}).")
        else:
            df[col] = df[col].fillna("UNKNOWN")
            log.append(f"Filled {n_missing:,} missing values in '{col}' with 'UNKNOWN'.")

    log.append(f"Rows kept: {len(df):,} out of {rows_before:,} ({len(df) / rows_before * 100:.2f}%).")
    return df, log


# ---------------------------------------------------------------------------
# 2. Feature engineering (temporal)
# ---------------------------------------------------------------------------
def add_time_features(df, date_column="Date"):
    """
    Derive Year, Month, Hour, DayOfWeek (and readable name columns) from Date.

    If the Date column is missing, the dataframe is returned unchanged so the
    rest of the app keeps working.
    """
    df = df.copy()
    if date_column not in df.columns:
        return df

    # The Chicago file uses "01/28/2025 03:30:00 PM". We give pandas that exact
    # format first (fast + safe) and fall back to automatic parsing if needed.
    parsed = pd.to_datetime(df[date_column], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(df[date_column], errors="coerce")

    df["Datetime"] = parsed
    df["Hour"] = parsed.dt.hour
    df["Month"] = parsed.dt.month
    df["Day"] = parsed.dt.day
    df["DayOfWeek"] = parsed.dt.dayofweek          # 0 = Monday ... 6 = Sunday
    df["DayName"] = parsed.dt.dayofweek.map(lambda d: DAY_NAMES[int(d)] if pd.notna(d) else "UNKNOWN")
    df["MonthName"] = parsed.dt.month.map(lambda m: MONTH_NAMES[int(m) - 1] if pd.notna(m) else "UNKNOWN")

    # Only create Year from the date if the dataset does not already have it.
    if "Year" not in df.columns:
        df["Year"] = parsed.dt.year

    # A handful of rows may have an unparsable date - fill those few gaps.
    for col in ["Hour", "Month", "Day", "DayOfWeek"]:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].astype(int)

    return df


def add_target(df, target_column="Arrest"):
    """Convert the True/False arrest flag into 1 = Arrest, 0 = No Arrest."""
    df = df.copy()
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in the dataset.")

    values = df[target_column]
    if values.dtype == bool:
        df["ArrestFlag"] = values.astype(int)
    else:
        # Handles the case where the CSV stored the flag as text
        df["ArrestFlag"] = (
            values.astype(str).str.strip().str.upper().isin(["TRUE", "1", "Y", "YES"]).astype(int)
        )
    return df


# ---------------------------------------------------------------------------
# 3. Feature selection + scikit-learn preprocessing
# ---------------------------------------------------------------------------
# Wish-list of features. Anything not present in the CSV is simply skipped.
CANDIDATE_CATEGORICAL = [
    "Primary Type",         # what kind of crime it is
    "Description",          # more detailed description of the offence
    "Location Description", # street, apartment, sidewalk, ...
    "District",             # police district
    "Ward",
    "Community Area",
    "Beat",                 # smallest police area
    "Domestic",             # domestic-related incident flag
]

CANDIDATE_NUMERICAL = [
    "Year",
    "Month",
    "Hour",
    "DayOfWeek",
]


def select_features(df):
    """
    Return the feature matrix X, the target y and the two feature-name lists.

    Only columns that actually exist in the dataframe are used, and none of the
    leakage/identifier columns from EXCLUDED_COLUMNS can end up in X.
    """
    categorical = [c for c in CANDIDATE_CATEGORICAL
                   if c in df.columns and c not in EXCLUDED_COLUMNS]
    numerical = [c for c in CANDIDATE_NUMERICAL
                 if c in df.columns and c not in EXCLUDED_COLUMNS]

    X = df[categorical + numerical].copy()

    # District / Ward / Beat are stored as numbers but they are really labels
    # (district 12 is not "twice" district 6), so we treat them as text.
    for col in categorical:
        X[col] = X[col].astype(str)

    y = df["ArrestFlag"]
    return X, y, categorical, numerical


def build_preprocessor(categorical, numerical):
    """
    ColumnTransformer used by both models.

    Numerical  : median imputation  -> standard scaling
    Categorical: most-frequent imputation -> one-hot encoding
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical),
            ("categorical", categorical_pipeline, categorical),
        ]
    )
    return preprocessor


def prepare_dataframe(df):
    """Convenience wrapper: clean -> time features -> 0/1 target."""
    clean_df, log = clean_data(df)
    clean_df = add_time_features(clean_df)
    clean_df = add_target(clean_df)
    return clean_df, log
