import numpy as np
import pandas as pd


def standardize_series(series):
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)

    if not np.isfinite(std) or std == 0:
        return values - mean

    return (values - mean) / std


def get_knn_recommendations(df, query_salary, k=3):
    if "salary" not in df.columns:
        return pd.DataFrame()

    if not np.isfinite(query_salary) or query_salary <= 0:
        return pd.DataFrame()

    if not isinstance(k, (int, np.integer)) or k < 1:
        return pd.DataFrame()

    salaries = pd.to_numeric(df["salary"], errors="coerce")
    valid_mask = np.isfinite(salaries) & (salaries > 0)

    valid_df = df.loc[valid_mask].copy()

    if valid_df.empty:
        return valid_df

    valid_salaries = salaries.loc[valid_mask]

    mean = valid_salaries.mean()
    std = valid_salaries.std(ddof=0)

    if not np.isfinite(std) or std == 0:
        std = 1.0

    # Manual one-dimensional Euclidean distance
    query_scaled = (query_salary - mean) / std
    salary_scaled = (valid_salaries - mean) / std

    valid_df["distance"] = np.abs(
        salary_scaled - query_scaled
    )

    return valid_df.nsmallest(min(k, len(valid_df)), "distance")


def predict_job_level(nearest_jobs):
    if "job_level" not in nearest_jobs.columns:
        return None

    labels = nearest_jobs["job_level"].dropna()
    labels = labels.astype(str).str.strip()
    labels = labels[labels != ""]

    if labels.empty:
        return None

    # Majority vote; alphabetical tie-breaking
    counts = labels.value_counts()
    winners = counts[counts == counts.max()].index.tolist()

    return sorted(winners)[0]
