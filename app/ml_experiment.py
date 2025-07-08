# DS-Containers/app/ml_experiment.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_and_log_expenses(data_path):
    """
    Reads weekly expense data, performs basic analysis, trains a model,
    and logs results to MLflow.
    data_path is expected to be relative to the container's root, e.g., /data/weekly_expense_YYYY-MM-DD.csv
    """
    logging.info(f"Starting ML experiment for data: {data_path}")

    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded {len(df)} records from {data_path}")
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_path}")
        return

    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month

    weekly_total_expense = df['total_price'].sum()
    avg_price_per_item = df['total_price'].sum() / df['quantity'].sum()
    top_product_category = df['product'].value_counts().idxmax()
    num_suppliers = df['supplier'].nunique()

    summary_stats = {
        "weekly_total_expense": weekly_total_expense.round(2),
        "avg_price_per_item": avg_price_per_item.round(2),
        "top_product_category": top_product_category,
        "num_suppliers": num_suppliers,
        "num_records": len(df)
    }
    logging.info(f"Weekly Summary: {summary_stats}")

    features = ['quantity', 'unit_price']
    target = 'total_price'

    if len(df) < 2:
        logging.warning("Not enough data to train ML model. Skipping model training.")
        model = None
        rmse = None
        r2 = None
    else:
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        logging.info(f"RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

    week_start_date = data_path.split('_')[-1].replace('.csv', '')
    mlflow.set_experiment(f"Restaurant Expense Report - Week {week_start_date}")

    with mlflow.start_run() as run:
        mlflow.log_param("data_file", os.path.basename(data_path))
        mlflow.log_param("num_records_processed", len(df))
        mlflow.log_param("model_type", "LinearRegression" if model else "None (Insufficient Data)")

        if model:
            mlflow.log_metric("weekly_total_expense", weekly_total_expense)
            mlflow.log_metric("avg_price_per_item", avg_price_per_item)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
        else:
            mlflow.log_metric("weekly_total_expense", weekly_total_expense)
            mlflow.log_metric("avg_price_per_item", avg_price_per_item)

        with open("summary_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=4)
        mlflow.log_artifact("summary_stats.json")
        os.remove("summary_stats.json")

        if model:
            mlflow.sklearn.log_model(model, "expense_prediction_model")
            logging.info("ML model logged to MLflow.")

        logging.info(f"MLflow Run ID: {run.info.run_id}")

    logging.info(f"ML experiment for {data_path} finished and logged.")