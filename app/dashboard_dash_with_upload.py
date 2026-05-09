import os
import base64
import io

import pandas as pd
import joblib
import plotly.express as px

from dash import Dash, html, dcc, dash_table, Input, Output, State

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
# Adds src/ folder to Python import to allow make_predictions function to work
sys.path.append(SRC_DIR)

from predict import make_predictions


DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_with_predictions.csv")

# creates Dash app
app = Dash(__name__)
app.title = "E-commerce Behavior Dashboard"

# to load fallback dataset
def load_default_data():
    if os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_csv(DEFAULT_DATA_PATH)
    return pd.DataFrame()

# processes uploaded CSV files
def parse_uploaded_file(contents, filename):
    content_type, content_string = contents.split(",") # seperates metadata, actual encoded file content
    decoded = base64.b64decode(content_string) # converts Base64 back into raw CSV text
    # checks uploaded file type
    if filename.endswith(".csv"):
        return pd.read_csv(io.StringIO(decoded.decode("utf-8"))) # decodes to text, wrap text as file-like object, pandas reads it

    raise ValueError("Please upload a CSV file.") # throws an error for invalid file types


# dynamically builds dashboard content
def create_dashboard_layout(df):
    if df.empty:
        return html.Div("No data available.")

    total_visitors = len(df)

    if "Revenue" in df.columns:
        buyers = int(df["Revenue"].sum())
        conversion_rate = buyers / total_visitors * 100
    else:
        buyers = "N/A"
        conversion_rate = None

    avg_purchase_probability = df["purchase_probability"].mean() * 100

    cards = html.Div(
        className="cards",
        children=[
            html.Div([
                html.H4("Total Visitors"),
                html.H2(f"{total_visitors:,}")
            ], className="card"),

            html.Div([
                html.H4("Actual Buyers"),
                html.H2(str(buyers))
            ], className="card"),

            html.Div([
                html.H4("Conversion Rate"),
                html.H2(f"{conversion_rate:.2f}%" if conversion_rate is not None else "N/A")
            ], className="card"),

            html.Div([
                html.H4("Avg Purchase Probability"),
                html.H2(f"{avg_purchase_probability:.2f}%")
            ], className="card"),
        ]
    )

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

    charts = [
        dcc.Graph(figure=fig_segments),
        dcc.Graph(figure=fig_prob)
    ]

    if "Revenue" in df.columns:
        revenue_counts = df["Revenue"].value_counts().reset_index()
        revenue_counts.columns = ["Revenue", "count"]

        fig_revenue = px.pie(
            revenue_counts,
            names="Revenue",
            values="count",
            title="Revenue True vs False"
        )

        charts.append(dcc.Graph(figure=fig_revenue))

    behavior_columns = [
        "ProductRelated",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues"
    ]

    available_behavior_columns = [col for col in behavior_columns if col in df.columns]

    if available_behavior_columns:
        metrics = df.groupby("behavior_segment")[available_behavior_columns].mean().round(3).reset_index()

        charts.append(html.H2("Behavior Metrics by Segment"))

        charts.append(
            dash_table.DataTable(
                data=metrics.to_dict("records"),
                columns=[{"name": col, "id": col} for col in metrics.columns],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "8px"},
                style_header={"fontWeight": "bold"}
            )
        )

        if "PageValues" in df.columns:
            charts.append(
                dcc.Graph(
                    figure=px.box(
                        df,
                        x="behavior_segment",
                        y="PageValues",
                        title="Page Values by Segment"
                    )
                )
            )

        if "BounceRates" in metrics.columns:
            charts.append(
                dcc.Graph(
                    figure=px.bar(
                        metrics,
                        x="behavior_segment",
                        y="BounceRates",
                        title="Average Bounce Rate by Segment"
                    )
                )
            )

        if "ExitRates" in metrics.columns:
            charts.append(
                dcc.Graph(
                    figure=px.bar(
                        metrics,
                        x="behavior_segment",
                        y="ExitRates",
                        title="Average Exit Rate by Segment"
                    )
                )
            )

    preview_table = dash_table.DataTable(
        data=df.head(20).to_dict("records"),
        columns=[{"name": col, "id": col} for col in df.columns],
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={
            "textAlign": "left",
            "padding": "8px",
            "maxWidth": "180px",
            "overflow": "hidden",
            "textOverflow": "ellipsis"
        },
        style_header={"fontWeight": "bold"}
    )

    return html.Div([
        cards,

        html.H2("Dataset Preview"),
        preview_table,

        html.H2("Visitor Segments"),
        *charts
    ])


app.layout = html.Div(
    style={"padding": "30px", "fontFamily": "Arial"},
    children=[
        html.H1("E-commerce Visitor Behavior Dashboard"),

        html.Div([
            html.H2("Upload CSV for Prediction"),

            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    "Drag and drop or ",
                    html.A("select a CSV file")
                ]),
                style={
                    "width": "100%",
                    "height": "90px",
                    "lineHeight": "90px",
                    "borderWidth": "2px",
                    "borderStyle": "dashed",
                    "borderRadius": "10px",
                    "textAlign": "center",
                    "marginBottom": "20px"
                },
                multiple=False # single file only
            ),

            html.Button(
                "Use Default Processed Dataset",
                id="default-data-button",
                n_clicks=0,
                style={
                    "padding": "10px 18px",
                    "borderRadius": "8px",
                    "border": "1px solid #ccc",
                    "cursor": "pointer"
                }
            ),

            html.Div(id="upload-status", style={"marginTop": "15px", "fontWeight": "bold"})
        ]),

        html.Hr(),

        html.Div(id="dashboard-content")
    ]
)


@app.callback(
    Output("dashboard-content", "children"),
    Output("upload-status", "children"),
    Input("upload-data", "contents"),
    Input("default-data-button", "n_clicks"),
    State("upload-data", "filename")
)

# runs whenever input changes
def update_dashboard(contents, n_clicks, filename):
    ctx = __import__("dash").callback_context
    # runs on initial page load
    if not ctx.triggered:
        df = load_default_data()
        return create_dashboard_layout(df), "Showing default processed dataset."

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # loads default dataset
    if triggered_id == "default-data-button":
        df = load_default_data()
        return create_dashboard_layout(df), "Showing default processed dataset."
    
    # runs when CSV uploaded
    if triggered_id == "upload-data" and contents is not None:
        try:
            uploaded_df = parse_uploaded_file(contents, filename) # reads uploaded CSV

            predicted_df = make_predictions(uploaded_df) # runs ML prediction pipeline

            return (
                create_dashboard_layout(predicted_df),
                f"Predictions generated successfully for: {filename}"
            )

        # catches runtime errors
        except Exception as e:
            return html.Div(), f"Error processing uploaded file: {str(e)}"

    df = load_default_data()
    return create_dashboard_layout(df), "Showing default processed dataset."


if __name__ == "__main__":
    app.run(debug=True)