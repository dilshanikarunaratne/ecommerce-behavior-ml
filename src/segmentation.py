def assign_behavior_segment(row):
    prob = row["purchase_probability"]

    product_pages = row.get("ProductRelated", 0)
    product_duration = row.get("ProductRelated_Duration", 0)
    bounce_rate = row.get("BounceRates", 0)
    exit_rate = row.get("ExitRates", 0)
    page_value = row.get("PageValues", 0)

    if prob >= 0.80 and page_value > 0:
        return "High Intent Buyer"

    elif prob >= 0.50:
        return "Potential Buyer"

    elif product_pages >= 10 and product_duration > 300 and bounce_rate < 0.05:
        return "Engaged Browser"

    elif bounce_rate > 0.10 or exit_rate > 0.30:
        return "At-Risk Visitor"

    else:
        return "Low Intent Visitor"