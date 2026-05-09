import os
import pandas as pd
import streamlit as st
import plotly.express as px


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "data_with_predictions.csv")


st.set_page_config(
    page_title="E-commerce Behavior Dashboard",
    layout="wide"
)

st.title("E-commerce Visitor Behavior Dashboard")

df = pd.read_csv(DATA_PATH)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# KPI cards
total_visitors = len(df)
buyers = df["Revenue"].sum()
conversion_rate = buyers / total_visitors * 100
avg_purchase_probability = df["purchase_probability"].mean() * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Visitors", total_visitors)
col2.metric("Actual Buyers", int(buyers))
col3.metric("Conversion Rate", f"{conversion_rate:.2f}%")
col4.metric("Avg Purchase Probability", f"{avg_purchase_probability:.2f}%")

st.divider()

# Segment distribution
st.subheader("Visitor Segments")

segment_counts = df["behavior_segment"].value_counts().reset_index()
segment_counts.columns = ["behavior_segment", "count"] # Rename columns 

# Creates a bar chart for count of segments
fig_segments = px.bar(
    segment_counts,
    x="behavior_segment",
    y="count",
    title="Number of Visitors by Behavior Segment"
)

st.plotly_chart(fig_segments, use_container_width=True)

# Groups users by segment and calculates average purchase probability
avg_prob = df.groupby("behavior_segment")["purchase_probability"].mean().reset_index()

# Creates a bar chart segment purchase probability
fig_prob = px.bar(
    avg_prob,
    x="behavior_segment",
    y="purchase_probability",
    title="Average Purchase Probability by Segment"
)

st.plotly_chart(fig_prob, use_container_width=True)

st.divider()

# Revenue distribution
st.subheader("Revenue Distribution")

revenue_counts = df["Revenue"].value_counts().reset_index()
revenue_counts.columns = ["Revenue", "count"]

# creates pie chart for revenue distribution
fig_revenue = px.pie(
    revenue_counts,
    names="Revenue",
    values="count",
    title="Revenue True vs False"
)

st.plotly_chart(fig_revenue, use_container_width=True)

# Behavior analysis
st.subheader("Behavior Metrics by Segment")

# Group by metrics
metrics = df.groupby("behavior_segment")[[
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues"
]].mean().reset_index()

# Display table
st.dataframe(metrics)

# creates a boxplot for page value
fig_page_values = px.box(
    df,
    x="behavior_segment",
    y="PageValues",
    title="Page Values by Segment"
)

st.plotly_chart(fig_page_values, use_container_width=True)

# creates a bar chart for bounce rate
fig_bounce = px.bar(
    metrics,
    x="behavior_segment",
    y="BounceRates",
    title="Average Bounce Rate by Segment"
)

st.plotly_chart(fig_bounce, use_container_width=True)

# creates a bar chart for exit rate
fig_exit = px.bar(
    metrics,
    x="behavior_segment",
    y="ExitRates",
    title="Average Exit Rate by Segment"
)

st.plotly_chart(fig_exit, use_container_width=True)