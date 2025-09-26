import pandas as pd

def basic_stats(df: pd.DataFrame):
    return {
        "rows": len(df),
        "cols": list(df.columns),
        "nulls": df.isna().sum().to_dict()
    }
