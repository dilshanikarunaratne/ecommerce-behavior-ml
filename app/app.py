import os
import pandas as pd
import plotly.express as px
import plotly.io as pio
from flask import Flask, render_template


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_with_predictions.csv")


@app.route("/")
def dashboard():
    df = pd.read_csv(DATA_PATH)

    total_visitors = len(df)
    buyers = int(df["Revenue"].sum())
    conversion_rate = buyers / total_visitors * 100
    avg_purchase_probability = df["purchase_probability"].mean() * 100

    segment_counts = df["behavior_segment"].value_counts().reset_index()
    segment_counts.columns = ["behavior_segment", "count"]

    fig_segments = px.bar(
        segment_counts,
        x="behavior_segment",
        y="count",
        title="Number of Visitors by Behavior Segment"
    )

    avg_prob = df.groupby("behavior_segment")["purchase_probability"].mean().reset_index()

    fig_prob = px.bar(
        avg_prob,
        x="behavior_segment",
        y="purchase_probability",
        title="Average Purchase Probability by Segment"
    )

    revenue_counts = df["Revenue"].value_counts().reset_index()
    revenue_counts.columns = ["Revenue", "count"]

    fig_revenue = px.pie(
        revenue_counts,
        names="Revenue",
        values="count",
        title="Revenue True vs False"
    )

    metrics = df.groupby("behavior_segment")[[
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues"
    ]].mean().round(3).reset_index()

    fig_page_values = px.box(
        df,
        x="behavior_segment",
        y="PageValues",
        title="Page Values by Segment"
    )

    fig_bounce = px.bar(
        metrics,
        x="behavior_segment",
        y="BounceRates",
        title="Average Bounce Rate by Segment"
    )

    fig_exit = px.bar(
        metrics,
        x="behavior_segment",
        y="ExitRates",
        title="Average Exit Rate by Segment"
    )

    charts = {
        "segments": pio.to_html(fig_segments, full_html=False),
        "probability": pio.to_html(fig_prob, full_html=False),
        "revenue": pio.to_html(fig_revenue, full_html=False),
        "page_values": pio.to_html(fig_page_values, full_html=False),
        "bounce": pio.to_html(fig_bounce, full_html=False),
        "exit": pio.to_html(fig_exit, full_html=False),
    }

    return render_template(
        "dashboard.html",
        total_visitors=total_visitors,
        buyers=buyers,
        conversion_rate=f"{conversion_rate:.2f}%",
        avg_purchase_probability=f"{avg_purchase_probability:.2f}%",
        preview=df.head(10).to_html(classes="data-table", index=False),
        metrics=metrics.to_html(classes="data-table", index=False),
        charts=charts
    )


if __name__ == "__main__":
    app.run(debug=True)