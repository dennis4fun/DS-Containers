# Containerized Restaurant Expense Reporting with MLflow & Streamlit

This project demonstrates a comprehensive MLOps workflow, combining Docker containerization, MLflow for experiment tracking, synthetic data generation, machine learning analysis, and an interactive Streamlit dashboard for weekly reporting. It simulates a weekly automated pipeline for a restaurant's expense tracking.

## 🌟 Features

- End-to-End Containerization: All core services (MLflow Tracking Server, Streamlit Dashboard) run in isolated Docker containers, managed by Docker Compose.

- MLflow Experiment Tracking: Logs machine learning experiment parameters, metrics, and models to a PostgreSQL backend store.

- Synthetic Data Generation: A Python script (`data_generator.py`) creates realistic-looking weekly expense data for various restaurant products.

- Machine Learning Analysis: An ML experiment (`ml_experiment.py`) processes the weekly data, performs basic analysis, trains a simple Linear Regression model, and logs key insights and the model to MLflow.

- Interactive Reporting Dashboard: A Streamlit application (`streamlit_app.py`) connects to the MLflow server to visualize weekly expense trends and experiment details.

- Automated Workflow Simulation: Includes a GitHub Actions workflow (`weekly_report_workflow.yml`) to demonstrate how data generation and ML analysis can be scheduled and automated (simulated locally).

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

  - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

  - `Python & Conda`: For running local scripts (data generation, ML experiment) and the local Streamlit dashboard.

- Step-by-Step Guide

1. Clone the Repository (or create the files manually):

```bash
git clone <your-repo-url>
cd DS-Containers
```

2. Create the `.env` file:
   In the root of the `DS-COntainers` directory (next to `docker-compose.yml`), create a file named `.env`.

```bash
# DS-Containers/.env
MLFLOW_TRACKING_URI=file:///reports/mlruns
MLFLOW_ARTIFACT_URI=file:///reports/artifacts
```

_Security Note:_ Always add `.env` to your `.gitignore` file to prevent sensitive information from being committed to version control.

3. Create data and reports directories on your host:
   These are the host directories that will be bind-mounted to your Docker containers.

```bash
mkdir data
mkdir reports
```

4. Build and Start the Docker Containers:
   Open your terminal, navigate to the restaurant_expense_tracker directory, and run:

```bash
docker-compose up --build -d
```

- `--build`: Builds the Docker images (specifically Dockerfile.mlflow and Dockerfile.streamlit) if they don't exist or have changed.

- `-d`: Runs the containers in detached mode (in the background).

This command will:

- Build your custom `mlflow-server` and `streamlit-ui` Docker images.

- Start all three services: MLflow Tracking Server, and the Streamlit UI.

1. Verify Containers are Running:
   You can check the status of your containers:

```bash
docker ps
```

You should see mlflow_postgres_db, mlflow_tracking_server, and restaurant_streamlit_ui listed with "Up" status.

6. Access the UIs:

- MLflow UI: Open your web browser and go to: http://localhost:5000

  - You should see the MLflow UI, initially empty.

- Streamlit UI: Open your web browser and go to: http://localhost:8504

  - You should see the Streamlit dashboard, which will initially show no runs.

1. Generate Data (Local Execution):
   This step creates the CSV data on your host, which ml_experiment.py will then read.

   - Open a PowerShell terminal and navigate to DS-Containers.

- Ensure your Conda environment (rag_env) is activated.

- Run:

```bash
python app/data_generator.py data
```

- You should see output confirming the CSV generation (e.g., Generated ... to: data\expense_data_YYYYMMDDHHMMSS.csv), and a new CSV should appear in DS-Containers/data/.

1. Run the ML Experiment (Local Execution, Logging to Container):
   This step processes the locally generated data and logs it to the containerized MLflow server.

   - In the same PowerShell terminal (still in DS-Containers):

```bash
# Get the name of the most recently generated CSV file for the experiment
$LATEST_CSV = Get-ChildItem -Path .\data -Filter "expense_data_*.csv" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty Name
$DATA_FILE_PATH = "data\" + $LATEST_CSV

Write-Host "Running ML experiment with data: $DATA_FILE_PATH"
python app/ml_experiment.py $DATA_FILE_PATH
```

- Observe the output carefully. You should see messages indicating data loading, analysis, and MLflow logging, without errors.

9. Verify Reports on Host:

   - Check your host's DS-Containers/reports/mlruns/ folder. You must see experiment folders created here, containing meta.yaml and run data.

   - Inside a run's artifacts folder (e.g., DS-Containers/reports/mlruns/<exp_id>/<run_id>/artifacts/), you should find raw_expense_data/ containing your generated CSV and summary_stats.json.

10. _Refresh the UIs:_

    - MLflow UI: http://localhost:5000 (Hard refresh: Ctrl + Shift + R or Ctrl + F5). You should now see your logged experiment data.

    - Streamlit UI: http://localhost:8504 (Hard refresh: Ctrl + Shift + R or Ctrl + F5). The dashboard should now display the aggregated results and trends.

11. View Reports and Trends:

    - Refresh your MLflow UI (http://localhost:5000): You should now see new experiments and runs logged, complete with parameters, metrics, and the saved model.

    - Refresh your Streamlit UI (http://localhost:8501): The dashboard will now display the aggregated results and trends from your MLflow runs.

## 📅 Simulating Weekly Automation (GitHub Actions)

The `.github/workflows/weekly_report_workflow.yml` file is included to demonstrate how this process could be automated in a real CI/CD environment like GitHub Actions.

## Important Note for Local Demo:

GitHub Actions runs in GitHub's cloud environment, not directly on your local machine. Therefore, the `weekly_report_workflow.yml` cannot directly connect to your local Docker Compose setup.

- For a true end-to-end GitHub Actions integration, your MLflow Tracking Server and Artifact Store would typically need to be deployed to a cloud service (e.g., MLflow on an AWS EC2 instance with S3 for artifacts, or Azure Machine Learning).

- For this local demo, the docker exec commands in Step 6 effectively simulate the automated weekly job that a CI/CD system would perform. You can run these commands manually each "Friday" to see new reports appear.

## 🧹 Cleaning Up

To stop and remove all containers, networks, and the persistent data volumes created by Docker Compose:

```bash
docker-compose down -v
```

- `down`: Stops and removes the containers and networks defined in docker-compose.yml.

- `-v`: Removes the named volumes (`mlflow_postgres_data`, `mlflow_tracking_store`, and `mlflow_artifacts`), mlflow_artifacts), which will delete all your database data and logged artifacts. Omit `-v` if you want to keep the data for future runs.

This project provides a robust and practical example of containerization in a data science context. Enjoy building and showcasing it!
