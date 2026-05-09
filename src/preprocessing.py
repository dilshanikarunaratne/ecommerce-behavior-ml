import pandas as pd

def preprocess_data(df):
    df = df.copy()

    
    # 1. Handle Target vaiable (Revenue) 
    # When you upload a CSV, the dataset does not neccessarily have the 'Revenue' column. Therefore, it is only considered if available
    
    if "Revenue" in df.columns:
        df["Revenue"] = df["Revenue"].astype(int)

    
    # 2. Handle missing values
    

    # Numerical columns, fill NA with median
    numerical_cols = [
        "Administrative", "Administrative_Duration",
        "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration",
        "BounceRates", "ExitRates",
        "PageValues", "SpecialDay"
    ]

    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Categorical columns, fill NA with "Unknown"
    categorical_cols = [
        "Month", "VisitorType", "OperatingSystems",
        "Browser", "Region", "TrafficType"
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # Boolean column, fill NA with "False"
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].fillna(False)

  
    # 3. One-hot Encoding
    
    if "VisitorType" in df.columns:
        df["VisitorType"] = df["VisitorType"].map({
            "Returning_Visitor": 1,
            "New_Visitor": 0,
            "Other": 0,
            "Unknown": 0
        })

    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)

    if "Month" in df.columns:
        df = pd.get_dummies(df, columns=["Month"], drop_first=True)

    return df