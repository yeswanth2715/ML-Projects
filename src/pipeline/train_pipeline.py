import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

def start_training_pipeline():
    try:
        logging.info("Training pipeline started.")

        # Step 1: Data ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed.")

        # Step 2: Data transformation
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path, test_path)
        logging.info("Data transformation completed.")

        # Step 3: Model training
        trainer = ModelTrainer()
        trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        logging.info("Model training completed.")

        logging.info("Training pipeline finished successfully.")

    except Exception as e:
        logging.error("Error in training pipeline.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    start_training_pipeline()

