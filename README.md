# Containerized Restaurant Expense Reporting with MLflow & Streamlit

This project demonstrates a comprehensive MLOps workflow, combining Docker containerization, MLflow for experiment tracking, synthetic data generation, machine learning analysis, and an interactive Streamlit dashboard for weekly reporting. It simulates a weekly automated pipeline for a restaurant's expense tracking.

## 🌟 Features

- End-to-End Containerization: All core services (PostgreSQL database, MLflow Tracking Server, Streamlit Dashboard) run in isolated Docker containers, managed by Docker Compose.

- MLflow Experiment Tracking: Logs machine learning experiment parameters, metrics, and models to a PostgreSQL backend store.

- Synthetic Data Generation: A Python script (data_generator.py) creates realistic-looking weekly expense data for various restaurant products.

- Machine Learning Analysis: An ML experiment (ml_experiment.py) processes the weekly data, performs basic analysis, trains a simple Linear Regression model, and logs key insights and the model to MLflow.

- Interactive Reporting Dashboard: A Streamlit application (streamlit_app.py) connects to the MLflow server to visualize weekly expense trends and experiment details.

- Automated Workflow Simulation: Includes a GitHub Actions workflow (weekly_report_workflow.yml) to demonstrate how data generation and ML analysis can be scheduled and automated (simulated locally).

- Data Persistence: Docker volumes ensure that your PostgreSQL database data and MLflow artifacts (models, reports) persist across container restarts.

## 🏗️ Project Structure

```Bash
DS-Containers/
├── .env                                # Environment variables (DB credentials, MLflow URI)
├── Dockerfile.mlflow                   # Dockerfile for the MLflow server container
├── Dockerfile.streamlit                # Dockerfile for the Streamlit UI container
├── docker-compose.yml                  # Orchestrates all Docker services
├── .github/                            # GitHub Actions workflows
│   └── workflows/
│       └── weekly_report_workflow.yml  # GitHub Actions workflow for weekly job
├── app/                                # Contains Python application code
│   ├── data_generator.py               # Generates synthetic weekly expense data
│   ├── ml_experiment.py                # ML analysis script (reads data, logs to MLflow)
│   ├── streamlit_app.py                # Streamlit UI for viewing reports
│   └── requirements.txt                # Consolidated Python dependencies for all app components
├── data/                               # Host directory for raw data (mounted as volume)
│   └── # weekly_expense_YYYY-MM-DD.csv files will be stored here
├── reports/                            # Host directory for MLflow artifacts (mounted as volume)
│   └── # MLflow artifacts (models, plots, summary_stats.json) will be stored here
└── README.md                           # This README file                       # This README file
```

## 🚀 Setup and Running the Project

- Prerequisites

  - Docker Desktop: Ensure Docker Desktop is installed and running on your machine.

- Download Docker Desktop

- Step-by-Step Guide

1. Clone the Repository (or create the files manually):

```bash
git clone <your-repo-url>
cd restaurant_expense_tracker
```

2. Create the .env file:
   In the root of the restaurant_expense_tracker directory (next to docker-compose.yml), create a file named .env. This file will store your database credentials.

```bash
# restaurant_expense_tracker/.env
# PostgreSQL Database Credentials for MLflow Backend Store
POSTGRES_DB=mlflow_db
POSTGRES_USER=mlflow_user
POSTGRES_PASSWORD=mlflow_password_secure

# MLflow Tracking URI (used by ml_experiment.py and Streamlit)
# This points to the 'mlflow-server' service within the Docker network
MLFLOW_TRACKING_URI=http://mlflow-server:5000
# MLflow Artifact Store URI (points to a local path within the container, mapped to a host volume)
MLFLOW_ARTIFACT_URI=file:/mlflow_artifacts
```

Security Note: Always add .env to your .gitignore file to prevent sensitive information from being committed to version control.

3. Build and Start the Docker Containers:
   Open your terminal, navigate to the restaurant_expense_tracker directory, and run:

```bash
docker-compose up --build -d
```

- --build: Builds the Docker images (specifically Dockerfile.mlflow and Dockerfile.streamlit) if they don't exist or have changed.

- -d: Runs the containers in detached mode (in the background).

This command will:

- Pull the postgres:13 Docker image.

- Build your custom mlflow-server and streamlit-ui Docker images.

- Start all three services: PostgreSQL database, MLflow Tracking Server, and the Streamlit UI.

4. Verify Containers are Running:
   You can check the status of your containers:

```bash
docker ps
```

You should see mlflow_postgres_db, mlflow_tracking_server, and restaurant_streamlit_ui listed with "Up" status.

5. Access the UIs:

- MLflow UI: Open your web browser and go to: http://localhost:5000

  - You should see the MLflow UI, initially empty.

- Streamlit UI: Open your web browser and go to: http://localhost:8501

  - You should see the Streamlit dashboard, which will initially show no runs.

6. Run the Machine Learning Experiment (Simulate Weekly Report):
   To populate MLflow with initial data, you'll execute the ml_experiment.py script inside the running mlflow_tracking_server container. This simulates your weekly automated job.

```bash
# Get the current week's Monday date (UTC)
WEEK_START_DATE=$(date -u +%Y-%m-%d -d "last monday")

# Execute the data generation and ML experiment inside the MLflow container
docker exec mlflow_tracking_server python app/data_generator.py ${WEEK_START_DATE}
docker exec mlflow_tracking_server python app/ml_experiment.py data/weekly_expense_${WEEK_START_DATE}.csv
```

- Explanation: The data_generator.py script will create a CSV file in the data/ volume, and then ml_experiment.py will read that CSV, run the analysis, and log to MLflow.

7. View Reports and Trends:

- Refresh your MLflow UI (http://localhost:5000): You should now see new experiments and runs logged, complete with parameters, metrics, and the saved model.

- Refresh your Streamlit UI (http://localhost:8501): The dashboard will now display the aggregated results and trends from your MLflow runs.

## 📅 Simulating Weekly Automation (GitHub Actions)

The .github/workflows/weekly_report_workflow.yml file is included to demonstrate how this process could be automated in a real CI/CD environment like GitHub Actions.

## Important Note for Local Demo:

GitHub Actions runs in GitHub's cloud environment, not directly on your local machine. Therefore, the weekly_report_workflow.yml cannot directly connect to your local Docker Compose setup.

- For a true end-to-end GitHub Actions integration, your MLflow Tracking Server and Artifact Store would typically need to be deployed to a cloud service (e.g., MLflow on an AWS EC2 instance with S3 for artifacts, or Azure Machine Learning).

- For this local demo, the docker exec commands in Step 6 effectively simulate the automated weekly job that a CI/CD system would perform. You can run these commands manually each "Friday" to see new reports appear.

## 🧹 Cleaning Up

To stop and remove all containers, networks, and the persistent data volumes created by Docker Compose:

```bash
docker-compose down -v
```

- down: Stops and removes the containers and networks defined in docker-compose.yml.

- -v: Removes the named volumes (mlflow_postgres_data, mlflow_artifacts), which will delete all your database data and logged artifacts. Omit -v if you want to keep the data for future runs.

This project provides a robust and practical example of containerization in a data science context. Enjoy building and showcasing it!
