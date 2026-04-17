import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Ingestion

RAW_DIR = os.path.join(PROJECT_ROOT, "artifacts", "raw")
RAW_FILE_PATH = os.path.join(RAW_DIR, "raw.csv")
TRAIN_FILE_PATH = os.path.join(RAW_DIR, "train.csv")
TEST_FILE_PATH = os.path.join(RAW_DIR, "test.csv")

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


######################## DATA PROCESSING ########################

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "artifacts", "processed")
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_train.csv")
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, "processed_test.csv")


####################### MODEL TRAINING #################
MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "artifacts", "models", "lgbm_model.pkl")