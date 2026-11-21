import re
import pandas as pd

KEYWORDS = [
    "intern",
    "junior",
    "senior",
    "sr.",
    "manager",
    "director",
    "executive",
    "lead",
]


def extract_features(df):
    features = pd.DataFrame(index=df.index)

    features["salary"] = pd.to_numeric(
        df["salary"], errors="coerce"
    )

    titles = df["job_title"].fillna("").astype(str).str.lower()

    for keyword in KEYWORDS:
        pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
        features[f"title_{keyword}"] = (
            titles.str.contains(pattern, regex=True).astype(int)
        )

    return features
