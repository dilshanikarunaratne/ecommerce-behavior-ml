import os
import pandas as pd
from dash import Dash, html, dcc, dash_table
import plotly.express as px


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_with_predictions.csv")

df = pd.read_csv(DATA_PATH)

total_visitors = len(df)
buyers = df["Revenue"].sum()
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
]].mean().reset_index()

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

# creates Dash web app
app = Dash(__name__)

app.layout = html.Div(
    style={"padding": "30px", "fontFamily": "Arial"},
    children=[
        html.H1("E-commerce Visitor Behavior Dashboard"),

        html.H2("Key Metrics"),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(4, 1fr)",
                "gap": "20px",
                "marginBottom": "30px"
            },
            children=[
                html.Div([
                    html.H4("Total Visitors"),
                    html.H2(f"{total_visitors:,}")
                ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px"}),

                html.Div([
                    html.H4("Actual Buyers"),
                    html.H2(f"{int(buyers):,}")
                ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px"}),

                html.Div([
                    html.H4("Conversion Rate"),
                    html.H2(f"{conversion_rate:.2f}%")
                ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px"}),

                html.Div([
                    html.H4("Avg Purchase Probability"),
                    html.H2(f"{avg_purchase_probability:.2f}%")
                ], style={"padding": "20px", "border": "1px solid #ddd", "borderRadius": "10px"}),
            ]
        ),

        html.H2("Dataset Preview"),

        dash_table.DataTable(
            data=df.head(10).to_dict("records"),
            columns=[{"name": col, "id": col} for col in df.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"}
        ),

        html.Hr(),

        html.H2("Visitor Segments"),

        dcc.Graph(figure=fig_segments),
        dcc.Graph(figure=fig_prob),

        html.Hr(),

        html.H2("Revenue Distribution"),

        dcc.Graph(figure=fig_revenue),

        html.Hr(),

        html.H2("Behavior Metrics by Segment"),

        dash_table.DataTable(
            data=metrics.round(3).to_dict("records"),
            columns=[{"name": col, "id": col} for col in metrics.columns],
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"}
        ),

        dcc.Graph(figure=fig_page_values),
        dcc.Graph(figure=fig_bounce),
        dcc.Graph(figure=fig_exit),
    ]
)

if __name__ == "__main__":
    app.run(debug=True)