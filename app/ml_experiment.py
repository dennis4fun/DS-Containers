# mlflow_grocery_tracker/app/ml_experiment.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_grocery_data(num_samples=100):
    """Generates synthetic grocery expense data."""
    np.random.seed(42)
    
    data = {
        'product_category': np.random.choice(['Produce', 'Dairy', 'Meat', 'Bakery', 'Snacks', 'Beverages'], num_samples),
        'quantity': np.random.randint(1, 10, num_samples),
        'unit_price': np.random.uniform(1.0, 15.0, num_samples),
        'discount_applied': np.random.choice([0, 0.05, 0.1, 0.15], num_samples, p=[0.7, 0.15, 0.1, 0.05]),
        'is_organic': np.random.choice([0, 1], num_samples, p=[0.8, 0.2]),
        'customer_loyalty_points': np.random.randint(0, 500, num_samples)
    }
    df = pd.DataFrame(data)
    
    # Calculate base_price and then total_expense
    df['base_price'] = df['quantity'] * df['unit_price']
    df['total_expense'] = df['base_price'] * (1 - df['discount_applied'])
    
    # Add some noise and make it dependent on loyalty points
    df['total_expense'] = df['total_expense'] + (df['customer_loyalty_points'] * 0.05) + np.random.normal(0, 2, num_samples)
    
    return df

def run_experiment(alpha=0.5, l1_ratio=0.5):
    """
    Simulates an ML experiment for grocery expense prediction and logs to MLflow.
    We'll predict 'total_expense' based on numerical features.
    """
    logging.info("Starting ML experiment...")

    # Set MLflow tracking URI to the containerized server
    # This environment variable is set in docker-compose.yml
    # mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")) # Fallback for local testing

    # Generate data
    df = generate_grocery_data()
    
    # Select features and target
    # For simplicity, let's use numerical features only for Linear Regression
    features = ['quantity', 'unit_price', 'discount_applied', 'is_organic', 'customer_loyalty_points']
    target = 'total_expense'

    X = df[features]
    y = df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Start an MLflow run
    # The artifact_uri is also set via environment variable in docker-compose.yml
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("alpha", alpha) # Example parameter
        mlflow.log_param("l1_ratio", l1_ratio) # Example parameter

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        logging.info(f"RMSE: {rmse}")
        logging.info(f"R2 Score: {r2}")

        # Log model
        mlflow.sklearn.log_model(model, "linear_regression_model")
        logging.info("Model logged to MLflow.")

    logging.info("ML experiment finished and logged.")

if __name__ == "__main__":
    # You can run experiments with different parameters
    run_experiment(alpha=0.5, l1_ratio=0.5)
    run_experiment(alpha=0.8, l1_ratio=0.2)
