import os
import sys
import json
import pandas as pd
import pymongo
from src.exception.exception import CustomException
from dotenv import load_dotenv
import certifi

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_URI")
ca = certifi.where()

class DataExtraction:
    def __init__(self):
        try:
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            print("MongoDB connection established.")
        except pymongo.errors.ConnectionError as e:
            raise CustomException(f"Failed to connect to MongoDB: {str(e)}", sys)
    
    def csv_to_json_chunks(self, file_path, chunk_size=5000):
        try:
            total_rows = pd.read_csv(file_path).shape[0]
            
            for start_row in range(0, total_rows, chunk_size):
                chunk = pd.read_csv(file_path, skiprows=start_row, nrows=chunk_size, header=0)
                chunk.reset_index(drop=True, inplace=True)
                records = list(json.loads(chunk.T.to_json()).values())
                yield records
        except FileNotFoundError:
            raise CustomException(f"File not found at path: {file_path}", sys)
        except Exception as e:
            raise CustomException(f"Failed to read CSV file in chunks: {str(e)}", sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            db = self.mongo_client[database]
            coll = db[collection]
            result = coll.insert_many(records)
            return len(result.inserted_ids)
        except Exception as e:
            raise CustomException(f"Failed to insert data into MongoDB: {str(e)}", sys)

if __name__ == '__main__':
    FILE_PATHS = {
        "MoviesLinks": "notebook/datasets/links.csv",
        "Movies": "notebook/datasets/movies.csv",
        "MoviesRatings": "notebook/datasets/ratings.csv",
        "MoviesTags": "notebook/datasets/tags.csv"
    }
    
    DATABASE = "MLData"
    extobj = DataExtraction()
    
    for collection, file_path in FILE_PATHS.items():
        try:
            total_inserted = 0
            for records in extobj.csv_to_json_chunks(file_path):
                inserted_count = extobj.insert_data_mongodb(records, DATABASE, collection)
                total_inserted += inserted_count
                print(f"Inserted {inserted_count} records into {collection}. Total inserted: {total_inserted}")
            
            print(f"Successfully inserted {total_inserted} records into MongoDB collection: {collection}")
        
        except CustomException as e:
            print(f"Error inserting data for {collection}: {e}")
