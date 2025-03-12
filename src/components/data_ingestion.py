from src.entity.config_entity import DataIngestionConfig 
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception.exception import CustomException
from src.logging.logger import logging
import os
import sys
import pandas as pd
import pymongo
from dotenv import load_dotenv
import certifi
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_URI")
ca = certifi.where()


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            logging.info("Initialized DataIngestion with config: %s", data_ingestion_config)
        except Exception as e:
            logging.error("Error initializing DataIngestion: %s", str(e))
            raise CustomException(e, sys)

    def export_multiple_collections_as_dataframe(self):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            db = self.mongo_client[self.data_ingestion_config.database_name]
            
            dataframes = {}
            collections = self.data_ingestion_config.collections_name
            
            for collection_name in collections:
                logging.info("Fetching data from MongoDB collection: %s in database: %s", collection_name, db)
                collection = db[collection_name]
                
                mongo_data = list(collection.find(limit=100))
                
                if not mongo_data:
                    logging.warning("No data found in collection: %s", collection_name)
                    continue
                
                for doc in mongo_data:
                    doc.pop("_id", None)
                
                columns = list(mongo_data[0].keys())
                values = [list(doc.values()) for doc in mongo_data]
                df = pd.DataFrame(values, columns=columns)
                
                dataframes[collection_name] = df
                logging.info("DataFrame created for collection %s with shape: %s", collection_name, df.shape)
            
            return dataframes
        except Exception as e:
            logging.error("Error exporting collections as DataFrames: %s", str(e))
            raise CustomException(e, sys)

        except Exception as e:
            logging.error("Error exporting collection as DataFrame: %s", str(e))
            raise CustomException(e, sys)

    def export_data_into_feature_store(self, dataframe, collection_name):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            
            feature_store_file_path = feature_store_file_path + f"{collection_name}.csv"
                
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info("Saving DataFrame to feature store at %s", feature_store_file_path)
            dataframe.to_csv(feature_store_file_path, index=False, header=True, encoding="utf-8", sep=",")
            
            logging.info("Data successfully saved to feature store for collection: %s", collection_name)
        except Exception as e:
            logging.error("Error exporting data into feature store: %s", str(e))
            raise CustomException(e, sys)

        
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process...")
            
            dataframes = self.export_multiple_collections_as_dataframe()
            print(dataframes.items())
            for collection_name, dataframe in dataframes.items():
                self.export_data_into_feature_store(dataframe , collection_name)
                logging.info("Exported data from collection: %s", collection_name)
            
            logging.info("Data ingestion completed successfully.")
            return DataIngestionArtifact(feature_store_path=self.data_ingestion_config.feature_store_file_path,files_name=self.data_ingestion_config.collections_name)
        
        except Exception as e:
            logging.error("Error in data ingestion process: %s", str(e))
            raise CustomException(e, sys)
