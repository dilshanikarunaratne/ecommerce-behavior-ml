import pandas as pd

def preprocess_data(df):
    df = df.copy()

    df['Revenue'] = df['Revenue'].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)

    df = pd.get_dummies(
        df,
        columns=['Month', 'VisitorType'],
        drop_first=True
    )

    return df