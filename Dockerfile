# mlflow_grocery_tracker/Dockerfile
# Use a Python base image suitable for MLflow
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Install MLflow and its dependencies
# psycopg2-binary is needed for PostgreSQL connectivity
# scikit-learn, pandas, numpy are for the example ML experiment
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the ML experiment script into the container
COPY app/ml_experiment.py app/

# Expose the default MLflow UI port
EXPOSE 5000

# The command to run the MLflow UI server is defined in docker-compose.yml
# CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]
