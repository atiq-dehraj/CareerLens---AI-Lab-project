import numpy as np
import pandas as pd

def detect_zscore_outliers(df, column='salary', threshold=3.0):
    if column not in df.columns:
        return pd.DataFrame(), 0

    # Convert salaries to numeric and exclude invalid values
    salaries = pd.to_numeric(df[column], errors='coerce')
    valid_salaries = salaries[
        np.isfinite(salaries) & (salaries > 0)
    ]

    # Handle insufficient data
    if len(valid_salaries) < 2:
        return pd.DataFrame(), 0

    # Calculate mean and standard deviation
    mean = valid_salaries.mean()
    std = valid_salaries.std(ddof=0)

    # Avoid division by zero
    if std == 0:
        return pd.DataFrame(), 0

    # Calculate Z-scores
    z_scores = (valid_salaries - mean) / std

    # Identify outliers
    outlier_scores = z_scores[z_scores.abs() > threshold]

    # Preserve original job records
    outliers = df.loc[outlier_scores.index].copy()
    outliers['z_score'] = outlier_scores

    return outliers, len(outliers)


def detect_iqr_outliers(df, column='salary'):
    if column not in df.columns:
        return pd.DataFrame(), {}

    # Convert salary values to numeric
    salaries = pd.to_numeric(df[column], errors='coerce')

    # Exclude missing, infinite, zero and negative salaries
    valid_salaries = salaries[
        np.isfinite(salaries) & (salaries > 0)
    ]

    # Handle empty datasets
    if valid_salaries.empty:
        return pd.DataFrame(), {}

    # Calculate quartiles
    Q1 = np.percentile(valid_salaries, 25)
    Q2 = np.percentile(valid_salaries, 50)
    Q3 = np.percentile(valid_salaries, 75)

    # Calculate interquartile range
    IQR = Q3 - Q1

    # Calculate outlier boundaries
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR

    # Identify outliers using valid salaries only
    outlier_values = valid_salaries[
        (valid_salaries < lower_fence) |
        (valid_salaries > upper_fence)
    ]

    # Preserve original job records
    outliers = df.loc[outlier_values.index].copy()

    stats = {
        "Q1": Q1,
        "Median": Q2,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Fence": lower_fence,
        "Upper Fence": upper_fence
    }

    return outliers, stats
