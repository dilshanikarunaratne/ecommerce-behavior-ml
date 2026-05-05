import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Handle Revenue safely
    if "Revenue" in df.columns:
        df["Revenue"] = df["Revenue"].astype(int)

    # Example encoding (adjust based on your code)
    if "VisitorType" in df.columns:
        df["VisitorType"] = df["VisitorType"].map({
            "Returning_Visitor": 1,
            "New_Visitor": 0,
            "Other": 0
        })

    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)

    if "Month" in df.columns:
        df = pd.get_dummies(df, columns=["Month"], drop_first=True)

    return df