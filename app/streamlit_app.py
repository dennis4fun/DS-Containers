# DS-Containers/app/streamlit_app.py
import streamlit as st
import mlflow
import pandas as pd
import plotly.express as px
import os
import json

st.set_page_config(page_title="Restaurant Expense Tracker Dashboard", layout="wide")
st.title("📊 Restaurant Expense Tracker Dashboard")

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)
st.write(f"Connected to MLflow Tracking Server at: `{mlflow_tracking_uri}`")

@st.cache_data
def get_mlflow_runs():
    """Fetches all MLflow runs and their metrics/params."""
    client = mlflow.tracking.MlflowClient()

    # Get all experiment IDs
    experiments = client.search_experiments()
    experiment_ids = [exp.experiment_id for exp in experiments]

    if not experiment_ids:
        return pd.DataFrame() # Return empty if no experiments

    runs = client.search_runs(
        experiment_ids=experiment_ids, # Search across all experiments
        order_by=["attribute.start_time DESC"],
        max_results=100 # Adjust as needed for more runs
    )

    data = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "start_time": run.info.start_time,
            "experiment_name": run.data.tags.get("mlflow.runName", "Default Experiment"),
        }
        for param_key, param_value in run.data.params.items():
            run_data[f"param.{param_key}"] = param_value
        for metric_key, metric_value in run.data.metrics.items():
            run_data[f"metric.{metric_key}"] = metric_value

        try:
            # Download artifact to a temporary location
            temp_artifact_path = client.download_artifacts(run_id=run.info.run_id, path="summary_stats.json")
            with open(temp_artifact_path, 'r') as f:
                summary_stats = json.load(f)
            for k, v in summary_stats.items():
                run_data[f"summary.{k}"] = v
            # Clean up temporary artifact file
            os.remove(temp_artifact_path)
        except Exception as e:
            # print(f"Could not download summary_stats.json for run {run.info.run_id}: {e}")
            pass

        data.append(run_data)

    return pd.DataFrame(data)

df_runs = get_mlflow_runs()

if df_runs.empty:
    st.info("No MLflow runs found. Please run the ML experiment script first.")
else:
    st.subheader("All Experiment Runs")
    st.dataframe(df_runs)

    st.subheader("Weekly Expense Trends")

    if 'start_time' in df_runs.columns:
        df_runs['start_time'] = pd.to_datetime(df_runs['start_time'], unit='ms')

        if 'metric.weekly_total_expense' in df_runs.columns:
            fig_total_expense = px.line(
                df_runs.sort_values('start_time'),
                x='start_time',
                y='metric.weekly_total_expense',
                title='Weekly Total Expense Over Time',
                labels={'start_time': 'Week Start Date', 'metric.weekly_total_expense': 'Total Expense ($)'}
            )
            st.plotly_chart(fig_total_expense, use_container_width=True)

        if 'metric.avg_price_per_item' in df_runs.columns:
            fig_avg_price = px.line(
                df_runs.sort_values('start_time'),
                x='start_time',
                y='metric.avg_price_per_item',
                title='Average Price Per Item Over Time',
                labels={'start_time': 'Week Start Date', 'metric.avg_price_per_item': 'Avg Price Per Item ($)'}
            )
            st.plotly_chart(fig_avg_price, use_container_width=True)

        if 'summary.top_product_category' in df_runs.columns:
            st.subheader("Top Product Categories Per Week")
            top_products_df = df_runs[['start_time', 'summary.top_product_category']].dropna()
            if not top_products_df.empty:
                st.dataframe(top_products_df.sort_values('start_time'))
            else:
                st.info("No 'Top Product Category' data found in summary artifacts.")

    else:
        st.warning("Could not find 'start_time' column in MLflow runs for plotting trends.")

st.subheader("Raw MLflow Data")
st.write("For more detailed information, visit the MLflow UI at http://localhost:5000")