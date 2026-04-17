import os
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.train_ratio = self.config["train_ratio"]
        self.test_ratio = self.config["test_ratio"]
        
        # Create artifacts/raw directory if it doesn't exist
        os.makedirs(RAW_DIR, exist_ok=True)
        
        # Path to local dataset
        self.local_dataset_path = os.path.join(PROJECT_ROOT.parent, "DATASET", "Hotel Reservations.csv", "Hotel Reservations.csv")
        
        logger.info(f"Data Ingestion initialized")
        logger.info(f"Local dataset path: {self.local_dataset_path}")
        logger.info(f"Output directory: {RAW_DIR}")
        logger.info(f"Train/Test split ratio: {self.train_ratio}/{self.test_ratio}")

    def load_local_dataset(self):
        """Load dataset from local DATASET folder and save as raw.csv"""
        try:
            logger.info(f"Loading dataset from: {self.local_dataset_path}")
            
            # Check if local dataset exists
            if not os.path.exists(self.local_dataset_path):
                raise FileNotFoundError(f"Dataset not found at: {self.local_dataset_path}")
            
            # Read the dataset
            data = pd.read_csv(self.local_dataset_path)
            logger.info(f"Dataset loaded successfully with shape: {data.shape}")
            
            # Save as raw.csv
            data.to_csv(RAW_FILE_PATH, index=False)
            logger.info(f"Raw data saved to: {RAW_FILE_PATH}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error while loading local dataset: {e}")
            raise CustomException("Failed to load local dataset", e)
    
    def split_data(self, data):
        """Split data into train (80%) and test (20%) sets"""
        try:
            logger.info("Starting data splitting process")
            logger.info(f"Total records: {len(data)}")
            
            # Split data into train and test
            train_data, test_data = train_test_split(
                data, 
                train_size=self.train_ratio,
                test_size=self.test_ratio,
                random_state=42,
                shuffle=True
            )
            
            # Save train data
            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            logger.info(f"Train data saved to: {TRAIN_FILE_PATH}")
            logger.info(f"Train records: {len(train_data)} ({self.train_ratio*100}%)")
            
            # Save test data
            test_data.to_csv(TEST_FILE_PATH, index=False)
            logger.info(f"Test data saved to: {TEST_FILE_PATH}")
            logger.info(f"Test records: {len(test_data)} ({self.test_ratio*100}%)")
            
            logger.info("Data splitting completed successfully")
            
        except Exception as e:
            logger.error(f"Error while splitting data: {e}")
            raise CustomException("Failed to split data into train and test sets", e)
    
    def run(self):
        """Execute the complete data ingestion pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("Starting Data Ingestion Process")
            logger.info("=" * 60)
            
            # Step 1: Load local dataset and save as raw.csv
            data = self.load_local_dataset()
            
            # Step 2: Split into train and test sets
            self.split_data(data)
            
            logger.info("=" * 60)
            logger.info("Data Ingestion Completed Successfully!")
            logger.info("=" * 60)
            logger.info(f"Files created:")
            logger.info(f"  1. Raw data: {RAW_FILE_PATH}")
            logger.info(f"  2. Train data: {TRAIN_FILE_PATH}")
            logger.info(f"  3. Test data: {TEST_FILE_PATH}")
            logger.info("=" * 60)
            
        except CustomException as ce:
            logger.error(f"CustomException: {str(ce)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise CustomException("Data ingestion failed", e)

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
