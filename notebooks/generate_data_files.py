import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_data(n_rows, filename):
    df = pd.DataFrame({
        "Administrative": np.random.randint(0, 10, n_rows),
        "Administrative_Duration": np.random.uniform(0, 200, n_rows),
        "Informational": np.random.randint(0, 5, n_rows),
        "Informational_Duration": np.random.uniform(0, 150, n_rows),
        "ProductRelated": np.random.randint(1, 50, n_rows),
        "ProductRelated_Duration": np.random.uniform(10, 2000, n_rows),
        "BounceRates": np.random.uniform(0.0, 0.2, n_rows),
        "ExitRates": np.random.uniform(0.0, 0.5, n_rows),
        "PageValues": np.random.uniform(0, 100, n_rows),
        "SpecialDay": np.random.uniform(0, 1, n_rows),
        "Month": np.random.choice([
            "Jan","Feb","Mar","Apr","May","June",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ], n_rows),
        "OperatingSystems": np.random.randint(1, 4, n_rows),
        "Browser": np.random.randint(1, 10, n_rows),
        "Region": np.random.randint(1, 9, n_rows),
        "TrafficType": np.random.randint(1, 20, n_rows),
        "VisitorType": np.random.choice(["Returning_Visitor", "New_Visitor"], n_rows),
        "Weekend": np.random.choice([True, False], n_rows)
    })

    os.makedirs("data/test", exist_ok=True)
    df.to_csv(f"data/test/{filename}", index=False)
    print(f"Saved: data/test/{filename}")


# Generate 2 datasets
generate_data(50, "test_data_1.csv")
generate_data(75, "test_data_2.csv")