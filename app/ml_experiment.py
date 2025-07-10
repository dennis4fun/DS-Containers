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
    Reads expense data, performs basic analysis, trains a model,
    and logs results to MLflow.
    data_path is expected to be the path to the generated CSV file (e.g., /data/expense_data_XYZ.csv).
    """
    logging.info(f"Starting ML experiment for data: {data_path}")

    # Set MLflow tracking URI explicitly for local execution
    # This will connect to the containerized MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # FIX: REMOVED mlflow.set_artifact_uri()
    # This function is deprecated and no longer needed.
    # The artifact URI is configured on the MLflow server side (docker-compose.yml).
    # local_mlflow_artifacts_path = os.path.abspath(os.path.join(os.getcwd(), 'reports', 'artifacts'))
    # mlflow.set_artifact_uri("file://" + local_mlflow_artifacts_path)


    try:
        df = pd.read_csv(data_path)
        logging.info(f"Loaded {len(df)} records from {data_path}")
    except FileNotFoundError:
        logging.error(f"Data file not found: {data_path}")
        raise # Re-raise to ensure the execution fails if file not found

    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month

    weekly_total_expense = df['total_price'].sum()
    avg_price_per_item = df['total_price'].sum() / df['quantity'].sum()
    top_product_category = df['product'].value_counts().idxmax()
    num_suppliers = df['supplier'].nunique()

    summary_stats = {
        "total_expense": weekly_total_expense.round(2),
        "avg_price_per_item": avg_price_per_item.round(2),
        "top_product_category": top_product_category,
        "num_suppliers": num_suppliers,
        "num_records": len(df),
        "data_file_used": os.path.basename(data_path)
    }
    logging.info(f"Summary: {summary_stats}")

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

    current_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    mlflow.set_experiment(f"Expense Report - Run {current_timestamp}")

    with mlflow.start_run() as run:
        mlflow.log_param("input_data_file", os.path.basename(data_path))
        mlflow.log_param("num_records_processed", len(df))
        mlflow.log_param("model_type", "LinearRegression" if model else "None (Insufficient Data)")
        
        if model:
            mlflow.log_metric("total_expense", summary_stats["total_expense"])
            mlflow.log_metric("avg_price_per_item", summary_stats["avg_price_per_item"])
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2_score", r2)
        else:
            mlflow.log_metric("total_expense", summary_stats["total_expense"])
            mlflow.log_metric("avg_price_per_item", summary_stats["avg_price_per_item"])

        with open("summary_stats.json", "w") as f:
            json.dump(summary_stats, f, indent=4)
        mlflow.log_artifact("summary_stats.json")
        os.remove("summary_stats.json") # Clean up local summary file

        # Log the raw data CSV as an artifact
        mlflow.log_artifact(data_path, artifact_path="raw_expense_data")
        # No need to os.remove(data_path) here, as it's in /data, not /tmp

        if model:
            # FIX: Commented out mlflow.sklearn.log_model as Model Registry is not supported by file store.
            # logging.info("ML model logged to MLflow.")
            logging.warning("ML model logging (to Model Registry) skipped as file store does not support it.")
        
        logging.info(f"MLflow Run ID: {run.info.run_id}")

    logging.info(f"ML experiment finished and logged.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # If an argument is provided, assume it's the data file path
        analyze_and_log_expenses(sys.argv[1])
    else:
        logging.error("No data file path provided. Usage: python ml_experiment.py <path_to_expense_csv>")
        sys.exit(1) # Exit if no data file is provided
