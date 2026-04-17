# 🏨 Hotel Reservation Prediction - MLOps Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-orange)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-red)](https://mlflow.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Architecture](#project-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training Pipeline](#model-training-pipeline)
- [CI/CD Deployment](#cicd-deployment)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This is an end-to-end **MLOps project** that predicts hotel reservation cancellations using machine learning. The project implements a complete ML pipeline from data ingestion to model deployment, featuring automated CI/CD workflows with Jenkins, Docker containerization, and cloud deployment on Google Cloud Platform (GCP).

### Problem Statement
Hotel reservation cancellations can significantly impact revenue and resource planning. This system predicts whether a customer will cancel their reservation based on various booking features, enabling hotels to:
- Optimize room allocation
- Implement dynamic pricing strategies
- Improve customer service
- Reduce revenue loss from cancellations

## ✨ Features

### Machine Learning Pipeline
- **Automated Data Ingestion** from Google Cloud Storage
- **Advanced Data Preprocessing** with feature engineering
- **Class Imbalance Handling** using SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Selection** using Random Forest feature importance
- **Hyperparameter Tuning** with RandomizedSearchCV
- **Model Training** with LightGBM Classifier
- **MLflow Integration** for experiment tracking and model versioning
  - Local SQLite database (`src/mlflow.db`)
  - Experiment runs stored in `src/mlruns/`
  - Web UI for comparing experiments

### MLOps & DevOps
- **Dockerized Application** for consistent deployment
- **CI/CD Pipeline** with Jenkins
- **Cloud Deployment** on Google Cloud Run
- **Automated Model Training** during container build
- **Version Control** with Git
- **Logging & Monitoring** throughout the pipeline
- **Modular Pipeline Components** (`custom_run/` scripts)
  - Individual scripts for data ingestion, preprocessing, and training
  - Enables debugging and testing of specific pipeline stages

### Web Application
- **Flask-based Web Interface** for predictions
- **User-friendly Form** for input data
- **Real-time Predictions** with trained model
- **Modern Responsive Design** with custom CSS styling

### Development Tools
- **Interactive Helper Script** (`run.bat` for Windows)
  - Menu-driven interface for common tasks
  - Quick access to pipeline components
  - Built-in testing and logging utilities
  - GCP credentials configuration
- **Jupyter Notebook** for exploratory data analysis
- **Comprehensive Logging** with daily log files
- **Configuration Management** with YAML files

## 🏗️ Project Architecture

```
┌─────────────────┐
│  GCP Storage    │
│  (Raw Data)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Ingestion  │
│  - Download CSV │
│  - Train/Test   │
│    Split        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Processing │
│  - Encoding     │
│  - SMOTE        │
│  - Feature Sel. │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│  - LightGBM     │
│  - Hyperparam   │
│  - MLflow       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Flask App      │
│  (Predictions)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Docker Image   │
│  (GCR)          │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cloud Run      │
│  (Production)   │
└─────────────────┘
```

## 🛠️ Technology Stack

### Machine Learning & Data Science
- **Python 3.8-3.12** - Core programming language (tested with 3.12)
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **LightGBM** - Gradient boosting framework
- **Imbalanced-learn** - SMOTE for handling class imbalance
- **MLflow** - Experiment tracking and model registry
- **Seaborn** - Statistical data visualization

### Web Framework
- **Flask** - Web application framework
- **HTML/CSS** - Frontend interface

### Cloud & DevOps
- **Google Cloud Platform (GCP)**
  - Cloud Storage - Data storage
  - Container Registry (GCR) - Docker image registry
  - Cloud Run - Serverless deployment
- **Docker** - Containerization
- **Jenkins** - CI/CD automation

### Development Tools
- **PyYAML** - Configuration management
- **Joblib** - Model serialization (via scikit-learn)
- **Git** - Version control
- **Jupyter Notebook** - Interactive development and EDA

## 📁 Project Structure

```
Hotel Reservation Prediction/
│
├── src/                              # Source code modules
│   ├── __init__.py
│   ├── data_ingestion.py            # GCP data download & splitting
│   ├── data_preprocessing.py        # Feature engineering & SMOTE
│   ├── model_training.py            # LightGBM training & MLflow
│   ├── logger.py                    # Logging configuration
│   ├── custom_exception.py          # Custom exception handling
│   ├── mlflow.db                    # MLflow tracking database
│   └── mlruns/                      # MLflow experiment runs
│
├── pipeline/                         # Training pipeline
│   ├── __init__.py
│   └── training_pipeline.py         # End-to-end training workflow
│
├── config/                           # Configuration files
│   ├── __init__.py
│   ├── config.yaml                  # Data processing parameters
│   ├── paths_config.py              # File paths configuration
│   └── model_params.py              # Model hyperparameters
│
├── utils/                            # Utility functions
│   ├── __init__.py
│   └── common_functions.py          # Helper functions (read_yaml, etc.)
│
├── custom_run/                       # Individual component runners
│   ├── run_data_ingestion.py       # Standalone data ingestion script
│   ├── run_data_preprocessing.py   # Standalone preprocessing script
│   └── run_model_training.py       # Standalone training script
│
├── templates/                        # HTML templates
│   └── index.html                   # Web interface
│
├── static/                           # Static files (CSS, JS)
│   └── style.css                    # Modern responsive styling
│
├── artifacts/                        # Generated artifacts
│   ├── raw/                         # Raw data (train.csv, test.csv)
│   ├── processed/                   # Processed data
│   └── models/                      # Trained models (lgbm_model.pkl)
│
├── logs/                             # Application logs
│   └── log_YYYY-MM-DD.log          # Daily log files
│
├── gcp_credentials/                  # GCP service account keys
│   ├── .gitkeep                     # Keep directory in git
│   ├── credentials_template.json   # Template for credentials structure
│   └── *.json                       # Actual credentials (gitignored)
│
├── notebook/                         # Jupyter notebooks
│   ├── notebook.ipynb               # EDA and experimentation
│   └── train.csv                    # Sample data for notebook
│
├── DATASET/                          # Original dataset files
│   └── Hotel Reservations.csv
│
├── CI-CD Deployment Materials/       # Deployment guides
│   └── CI-CD Deployment on Jenkins.txt
│
├── PROJECT CODE/                     # Legacy directory
│   └── custom_jenkins/              # Jenkins Docker setup
│       └── Dockerfile               # Custom Jenkins with Docker-in-Docker
│
├── MLOPS_PROJECT_1.egg-info/        # Package metadata (auto-generated)
│
├── htlvenv/                          # Virtual environment (gitignored)
│
├── application.py                    # Flask web application
├── Dockerfile                        # Application Docker image
├── Jenkinsfile                       # CI/CD pipeline definition
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup configuration
├── run.bat                           # Windows helper script for running tasks
├── .gitignore                        # Git ignore rules
└── README.md                         # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 to 3.12 (tested with 3.12)
- Docker Desktop (for containerization)
- Google Cloud Platform account (for deployment)
- Git
- Jupyter Notebook (optional, for EDA)

### Local Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd "Hotel Reservation Prediction"
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv htlvenv
   
   # Windows
   htlvenv\Scripts\activate
   
   # Linux/Mac
   source htlvenv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. **Set Up GCP Credentials**
   
   ⚠️ **Important Security Note**: Never commit actual GCP credentials to Git!
   
   - Create a service account in your GCP project with **Storage Object Viewer** permissions
   - Download the JSON key file from GCP Console
   - Place it in `gcp_credentials/` directory
   - Use `credentials_template.json` as a reference for the required structure
   
   Set the environment variable:
   ```bash
   # Windows (PowerShell)
   $env:GOOGLE_APPLICATION_CREDENTIALS="gcp_credentials\your-credentials.json"
   
   # Windows (CMD)
   set GOOGLE_APPLICATION_CREDENTIALS=gcp_credentials\your-credentials.json
   
   # Linux/Mac
   export GOOGLE_APPLICATION_CREDENTIALS=gcp_credentials/your-credentials.json
   ```
   
   **Note**: The actual credentials file is automatically excluded by `.gitignore` to prevent accidental commits.

5. **Configure Settings**
   - Update `config/config.yaml` with your GCP bucket details
   - Modify `config/paths_config.py` if needed

## 💻 Usage

### Quick Start with run.bat (Windows)

For Windows users, the project includes a convenient helper script:

```bash
run.bat
```

This interactive menu provides the following options:
1. **Run Complete Training Pipeline** - Execute the full end-to-end pipeline
2. **Run Data Ingestion Only** - Download and split data from GCP
3. **Run Data Preprocessing Only** - Process and transform data
4. **Run Model Training Only** - Train the LightGBM model
5. **Start Web Application** - Launch the Flask web interface
6. **Test Imports** - Verify all modules are correctly installed
7. **View Latest Logs** - Display recent log entries
8. **Set GCP Credentials** - Configure GCP credentials for the session
9. **Exit** - Close the helper script

### Training the Model

**Option 1: Using the helper script (Windows)**
```bash
run.bat
# Select option 1
```

**Option 2: Direct execution**
```bash
python pipeline/training_pipeline.py
```

**Option 3: Individual components**
```bash
# Run each step separately
python custom_run/run_data_ingestion.py
python custom_run/run_data_preprocessing.py
python custom_run/run_model_training.py
```

The training pipeline will:
1. Download data from GCP Storage
2. Split into train/test sets
3. Preprocess and engineer features
4. Apply SMOTE for class balancing
5. Select top features using Random Forest
6. Train LightGBM model with hyperparameter tuning
7. Log experiments to MLflow
8. Save the trained model

### Running the Web Application

**Option 1: Using the helper script (Windows)**
```bash
run.bat
# Select option 5
```

**Option 2: Direct execution**
```bash
python application.py
```

Access the application at: `http://localhost:8080`

### Using Docker

1. **Build Docker Image**
   ```bash
   docker build -t hotel-reservation-prediction .
   ```

2. **Run Container**
   ```bash
   docker run -d -p 8080:8080 hotel-reservation-prediction
   ```

## 🔄 Model Training Pipeline

### 1. Data Ingestion (`src/data_ingestion.py`)
- Downloads `Hotel_Reservations.csv` from GCP Storage bucket
- Splits data into 80% training and 20% testing sets
- Saves raw data to `artifacts/raw/`

### 2. Data Preprocessing (`src/data_preprocessing.py`)

**Preprocessing Steps:**
- Drop unnecessary columns (`Unnamed: 0`, `Booking_ID`)
- Remove duplicate records
- **Label Encoding** for categorical features:
  - `type_of_meal_plan`
  - `required_car_parking_space`
  - `room_type_reserved`
  - `market_segment_type`
  - `repeated_guest`
  - `booking_status` (target variable)

**Skewness Handling:**
- Apply log transformation to features with skewness > 5
- Numerical features processed:
  - Lead time, number of adults/children
  - Weekend/weekday nights
  - Arrival details (year, month, date)
  - Previous cancellations/bookings
  - Average price per room
  - Special requests

**Class Imbalance Handling:**
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique)
- Balance the target variable distribution
- Prevents model bias towards majority class

**Feature Selection:**
- Train Random Forest Classifier
- Calculate feature importance scores
- Select **top 10 features**:
  1. Lead time
  2. Number of special requests
  3. Average price per room
  4. Arrival month
  5. Arrival date
  6. Market segment type
  7. Number of week nights
  8. Number of weekend nights
  9. Type of meal plan
  10. Room type reserved

### 3. Model Training (`src/model_training.py`)

**Algorithm:** LightGBM Classifier

**Hyperparameter Tuning:**
- Method: RandomizedSearchCV
- Cross-validation: 5-fold
- Iterations: 10
- Scoring metric: Accuracy

**Hyperparameters Tuned:**
- `n_estimators`: Number of boosting iterations
- `max_depth`: Maximum tree depth
- `learning_rate`: Step size shrinkage
- `num_leaves`: Maximum leaves in one tree
- `min_child_samples`: Minimum data in leaf
- `subsample`: Fraction of data for training
- `colsample_bytree`: Fraction of features for training

**Evaluation Metrics:**
- Accuracy Score
- Precision Score
- Recall Score
- F1 Score

**MLflow Integration:**
- Logs training/testing datasets
- Tracks hyperparameters
- Records evaluation metrics
- Saves trained model artifacts
- Enables experiment comparison

### 4. Modular Execution with custom_run/

The project includes standalone scripts for running individual pipeline components:

**`custom_run/run_data_ingestion.py`**
- Downloads data from GCP Storage
- Performs train-test split
- Saves raw data to artifacts directory
- Useful for testing data ingestion independently

**`custom_run/run_data_preprocessing.py`**
- Loads raw data from artifacts
- Applies preprocessing transformations
- Handles class imbalance with SMOTE
- Performs feature selection
- Saves processed data

**`custom_run/run_model_training.py`**
- Loads processed data
- Trains LightGBM model with hyperparameter tuning
- Logs experiments to MLflow
- Saves trained model to artifacts

These scripts allow you to:
- Debug individual pipeline stages
- Re-run specific components without full pipeline execution
- Test changes to individual modules
- Develop and iterate faster

### 5. MLflow Experiment Tracking

The project uses MLflow for comprehensive experiment tracking:

**MLflow Database:** `src/mlflow.db`  
**Experiment Runs:** `src/mlruns/`

**Tracked Information:**
- Model hyperparameters
- Training and validation metrics
- Dataset versions
- Model artifacts
- Run timestamps and duration

**View MLflow UI:**
```bash
cd src
mlflow ui
```
Access at: `http://localhost:5000`

This allows you to:
- Compare different model runs
- Analyze hyperparameter impact
- Track model performance over time
- Reproduce experiments
- Select the best performing model

## 🚢 CI/CD Deployment

### Jenkins Pipeline Stages

The `Jenkinsfile` defines a 4-stage CI/CD pipeline:

#### Stage 1: Clone Repository
```groovy
- Checkout code from GitHub
- Branch: main
- Uses GitHub token credentials
```

#### Stage 2: Setup Environment
```groovy
- Create Python virtual environment
- Upgrade pip
- Install project dependencies
```

#### Stage 3: Build & Push Docker Image
```groovy
- Authenticate with GCP
- Configure Docker for GCR
- Build Docker image
- Push to Google Container Registry
- Image: gcr.io/{PROJECT_ID}/ml-project:latest
```

#### Stage 4: Deploy to Cloud Run
```groovy
- Deploy container to Google Cloud Run
- Platform: Managed
- Region: us-central1
- Allow unauthenticated access
```

### Docker Configuration

**Dockerfile Highlights:**
- Base image: `python:3.12-slim`
- Environment variables:
  - `PYTHONDONTWRITEBYTECODE=1` - Prevents Python from writing .pyc files
  - `PYTHONUNBUFFERED=1` - Ensures logs are displayed in real-time
- Installs LightGBM system dependencies (`libgomp1`)
- Copies application code and requirements
- Installs Python packages with pip caching disabled for smaller image size
- Installs project in editable mode (`pip install -e .`)
- **Trains model during build** (ensures fresh model in container)
- Exposes port 8080
- Runs Flask application with `python application.py`

### Setting Up Jenkins

Detailed instructions available in `CI-CD Deployment Materials/STEPS.md`

**Quick Setup:**
1. Install Docker Desktop
2. Build custom Jenkins image with Docker-in-Docker
3. Run Jenkins container
4. Install Python, pip, and Google Cloud SDK
5. Configure GCP credentials in Jenkins
6. Create Jenkins pipeline from `Jenkinsfile`
7. Trigger build

## 📊 API Documentation

### Prediction Endpoint

**URL:** `/`  
**Method:** `POST`  
**Content-Type:** `application/x-www-form-urlencoded`

**Request Parameters:**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `lead_time` | integer | Days between booking and arrival | 120 |
| `no_of_special_request` | integer | Number of special requests | 2 |
| `avg_price_per_room` | float | Average room price | 85.50 |
| `arrival_month` | integer | Month of arrival (1-12) | 7 |
| `arrival_date` | integer | Date of arrival (1-31) | 15 |
| `market_segment_type` | integer | Market segment (0-4) | 4 |
| `no_of_week_nights` | integer | Number of weekday nights | 3 |
| `no_of_weekend_nights` | integer | Number of weekend nights | 2 |
| `type_of_meal_plan` | integer | Meal plan type (0-3) | 1 |
| `room_type_reserved` | integer | Room type (0-6) | 2 |

**Market Segment Types:**
- 0: Aviation
- 1: Complimentary
- 2: Corporate
- 3: Offline
- 4: Online

**Meal Plan Types:**
- 0: Meal Plan 1
- 1: Meal Plan 2
- 2: Meal Plan 3
- 3: Not Selected

**Room Types:**
- 0-6: Room Type 1 through Room Type 7

**Response:**

Returns HTML page with prediction result:
- **Prediction = 0**: "The Customer is going to cancel his reservation"
- **Prediction = 1**: "The Customer is not going to cancel his reservation"

**Example cURL Request:**
```bash
curl -X POST http://localhost:8080/ \
  -d "lead_time=120" \
  -d "no_of_special_request=2" \
  -d "avg_price_per_room=85.50" \
  -d "arrival_month=7" \
  -d "arrival_date=15" \
  -d "market_segment_type=4" \
  -d "no_of_week_nights=3" \
  -d "no_of_weekend_nights=2" \
  -d "type_of_meal_plan=1" \
  -d "room_type_reserved=2"
```

## 📈 Model Performance

The LightGBM model is evaluated on the following metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate
- **F1 Score**: Harmonic mean of precision and recall

All metrics are logged to MLflow for tracking and comparison across experiments.

## 🔧 Configuration

### `config/config.yaml`

```yaml
data_ingestion:
  bucket_name: "my_bucket_aiops"
  bucket_file_name: "Hotel_Reservations.csv"
  train_ratio: 0.8
  test_ratio: 0.2

data_processing:
  categorical_columns:
    - type_of_meal_plan
    - required_car_parking_space
    - room_type_reserved
    - market_segment_type
    - repeated_guest
    - booking_status
  numerical_columns:
    - no_of_adults
    - no_of_children
    - no_of_weekend_nights
    - no_of_week_nights
    - lead_time
    - arrival_year
    - arrival_month
    - arrival_date
    - no_of_previous_cancellations
    - no_of_previous_bookings_not_canceled
    - avg_price_per_room
    - no_of_special_requests
  skewness_threshold: 5
  no_of_features: 10
```

### `config/model_params.py`

Defines LightGBM hyperparameter search space and RandomizedSearchCV settings.

### `config/paths_config.py`

Centralizes all file paths for data, models, and artifacts.

## 🐛 Troubleshooting

### Common Issues

1. **GCP Authentication Error**
   - Ensure `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set
   - Verify service account has Storage Object Viewer permissions
   - Check that the credentials JSON file exists in `gcp_credentials/`

2. **Module Import Errors**
   - Run `pip install -e .` to install package in editable mode
   - Ensure virtual environment is activated (`htlvenv`)
   - Verify you're in the project root directory

3. **Docker Build Failures**
   - Check Docker Desktop is running
   - Verify Dockerfile syntax
   - Ensure sufficient disk space
   - Clear Docker cache: `docker system prune -a`

4. **Model Training Errors**
   - Verify data is downloaded to `artifacts/raw/`
   - Check GCP bucket name and file name in `config/config.yaml`
   - Ensure sufficient memory for SMOTE operation (at least 4GB RAM recommended)
   - Check logs in `logs/` directory for detailed error messages

5. **Jenkins Pipeline Failures**
   - Verify GCP credentials are configured in Jenkins
   - Check Docker permissions for Jenkins user
   - Ensure Google Cloud SDK is installed in Jenkins container
   - Review Jenkins console output for specific errors

6. **run.bat Helper Script Issues (Windows)**
   - Ensure you're running from the project root directory
   - Verify Python is in your system PATH
   - Check that all required files exist (pipeline/, custom_run/, etc.)
   - If GCP credentials option fails, manually set the environment variable

7. **MLflow UI Not Starting**
   - Navigate to `src/` directory before running `mlflow ui`
   - Check if port 5000 is already in use
   - Verify `mlflow.db` exists in the `src/` directory

8. **Flask Application Port Conflict**
   - If port 8080 is in use, modify `application.py` to use a different port
   - Check for other applications using port 8080
   - Use `netstat -ano | findstr :8080` (Windows) to identify the process

## 📝 Logging

The project implements comprehensive logging throughout the pipeline:

- **Logger Configuration**: `src/logger.py`
- **Log Location**: `logs/` directory
- **Log Format**: Timestamped with module name and log level
- **Custom Exceptions**: `src/custom_exception.py` for detailed error tracking

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Sudhanshu** - Initial work and MLOps implementation

## 🙏 Acknowledgments

- Hotel reservation dataset from Kaggle
- LightGBM documentation and community
- MLflow for experiment tracking
- Google Cloud Platform for deployment infrastructure
- Jenkins community for CI/CD best practices

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Note**: This project is designed for educational and demonstration purposes, showcasing end-to-end MLOps practices including data engineering, model training, containerization, and cloud deployment.
