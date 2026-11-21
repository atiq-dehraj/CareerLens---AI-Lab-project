import pandas as pd
import numpy as np


def clean_data(df):
    # Work on a copy of the original dataset
    df = df.copy()

    # Remove only records without job titles
    df = df.dropna(subset=['job_title']).copy()

    # Preserve records with missing company names
    df['company'] = df['company'].fillna('Unknown')

    # Convert salaries to numeric values
    if 'salary' in df.columns:
        df['salary'] = pd.to_numeric(
            df['salary'], errors='coerce'
        )

        # Keep only positive salary values
        df['salary'] = df['salary'].where(
            df['salary'] > 0
        )

    # Do not fill missing salaries with the median.
    # Do not remove duplicates without a reliable job ID.

    return df
