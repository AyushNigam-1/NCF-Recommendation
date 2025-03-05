from src.entity.config_entity import DataIngestionConfig 
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception.exception import CustomException
from src.logging.logger import logging
import os
import sys
import numpy as np
import pandas as pd
import pymongo
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("Initialized DataIngestion with config: %s", data_ingestion_config)
        except Exception as e:
            logging.error("Error initializing DataIngestion: %s", str(e))
            raise CustomException(e, sys)

    def export_collection_as_dataframe(self): 
        try:
            logging.info("Fetching data from MongoDB collection: %s in database: %s",
                         self.data_ingestion_config.collection_name, self.data_ingestion_config.database_name)

            # Connect to MongoDB
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[self.data_ingestion_config.database_name][self.data_ingestion_config.collection_name]

            # Fetch data
            mongo_data = list(collection.find().limit(1000))

            if not mongo_data:
                raise ValueError("No data found in the collection.")

            # Drop `_id` field from each document
            for doc in mongo_data:
                doc.pop("_id", None)

            # Extract column names and values
            columns = list(mongo_data[0].keys())
            values = [list(doc.values()) for doc in mongo_data]

            # Create DataFrame
            df = pd.DataFrame(values, columns=columns)

            # Convert 'na' strings to NaN
            df.replace({"na": np.nan}, inplace=True)

            logging.info("DataFrame created with shape: %s", df.shape)
            return df

        except Exception as e:
            logging.error("Error exporting collection as DataFrame: %s", str(e))
            raise CustomException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Ensure correct file extension
            if not feature_store_file_path.endswith(".csv"):
                feature_store_file_path = feature_store_file_path.rsplit(".", 1)[0] + ".csv"

            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Saving DataFrame to feature store at %s", feature_store_file_path)

            # Save DataFrame to CSV
            dataframe.to_csv(feature_store_file_path, index=False, header=True, encoding="utf-8", sep=",")

            logging.info("Data successfully saved to feature store.")
            return dataframe

        except Exception as e:
            logging.error("Error exporting data into feature store: %s", str(e))
            raise CustomException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process...")

            dataframe = self.export_collection_as_dataframe()
            self.export_data_into_feature_store(dataframe)

            logging.info("Data ingestion completed successfully.")
            return DataIngestionArtifact(feature_store_path=self.data_ingestion_config.feature_store_file_path)

        except Exception as e:
            logging.error("Error in data ingestion process: %s", str(e))
            raise CustomException(e, sys)
