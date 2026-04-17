#!/usr/bin/env python
"""
Standalone Model Training Script
This script can be run from anywhere and will work correctly.
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import ModelTraining
from config.paths_config import PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH

if __name__ == "__main__":
    print("=" * 60)
    print("Hotel Reservation Prediction - Model Training")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Train Data: {PROCESSED_TRAIN_DATA_PATH}")
    print(f"Test Data: {PROCESSED_TEST_DATA_PATH}")
    print(f"Model Output: {MODEL_OUTPUT_PATH}")
    print("=" * 60)
    
    trainer = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    trainer.run()
    
    print("=" * 60)
    print("Model Training Complete!")
    print("=" * 60)
