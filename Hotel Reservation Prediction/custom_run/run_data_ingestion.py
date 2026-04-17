#!/usr/bin/env python
import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from src.data_ingestion import DataIngestion
from utils.common_functions import read_yaml
from config.paths_config import CONFIG_PATH

if __name__ == "__main__":
    print("=" * 60)
    print("Hotel Reservation Prediction - Data Ingestion")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Config Path: {CONFIG_PATH}")
    print("=" * 60)
    
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
    
    print("=" * 60)
    print("Data Ingestion Complete!")
    print("=" * 60)
