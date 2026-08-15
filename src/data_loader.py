"""
data_loader.py
--------------
Loads the Chicago Crime CSV file and provides simple helper functions
used for the "Data Overview" and "Data Quality" parts of the project.

Everything here is plain pandas so that each step is easy to explain.
"""

import os
import pandas as pd

# Path of the dataset (data/chicago_crime.csv relative to the project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "chicago_crime.csv")

# The target column of the Machine Learning part
TARGET_COLUMN = "Arrest"


def load_raw_data(path=DATA_PATH, sample_size=None, random_state=42):
    """
    Read the CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Location of the CSV file.
    sample_size : int or None
        If given (and smaller than the dataset), a random sample of that many
        rows is returned. This keeps the Streamlit app fast on very large files.
    random_state : int
        Fixed seed so the same sample is drawn every time (reproducibility).

    Returns
    -------
    df : pandas.DataFrame
    info : dict with details about what was loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place the Chicago crime CSV at data/chicago_crime.csv"
        )

    df = pd.read_csv(path, low_memory=False)
    total_rows = len(df)

    # Optional sampling for performance (the user is always told about it)
    sampled = False
    if sample_size is not None and sample_size < total_rows:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        sampled = True

    info = {
        "total_rows_in_file": total_rows,
        "rows_loaded": len(df),
        "sampled": sampled,
        "path": path,
    }

    # The ML target must exist, otherwise we stop instead of guessing another one.
    if TARGET_COLUMN not in df.columns:
        raise KeyError(
            f"The required target column '{TARGET_COLUMN}' was not found in the dataset. "
            f"Columns available: {list(df.columns)}"
        )

    return df, info


def split_column_types(df):
    """
    Split the columns into numerical and categorical lists.

    We check the dtype directly instead of using select_dtypes so that the code
    behaves the same on both pandas 2.x and pandas 3.x (where text columns
    changed from 'object' to 'str').
    """
    numerical, categorical = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
            numerical.append(col)
        else:
            categorical.append(col)
    return numerical, categorical


def missing_value_report(df):
    """Return a small table with the missing count and percentage of every column."""
    missing_count = df.isna().sum()
    missing_percent = (missing_count / len(df)) * 100

    report = pd.DataFrame(
        {
            "Column": missing_count.index,
            "Missing Count": missing_count.values,
            "Missing %": missing_percent.round(2).values,
        }
    )
    return report.sort_values("Missing Count", ascending=False).reset_index(drop=True)


def duplicate_report(df, id_column="ID"):
    """Count fully duplicated rows and (if present) duplicated ID values."""
    report = {"duplicate_rows": int(df.duplicated().sum())}
    if id_column in df.columns:
        report["duplicate_ids"] = int(df[id_column].duplicated().sum())
    return report


def basic_overview(df):
    """A few headline numbers used on the Overview page of the app."""
    overview = {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if TARGET_COLUMN in df.columns:
        arrest = df[TARGET_COLUMN].astype(bool)
        overview["arrest_rate"] = float(arrest.mean() * 100)
        overview["arrest_count"] = int(arrest.sum())

    if "Primary Type" in df.columns:
        overview["crime_types"] = int(df["Primary Type"].nunique())

    return overview
