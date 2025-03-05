import pandas as pd
import numpy as np
import os
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from src.constants.training_pipeline import SCHEMA_FILE_PATH

class DataCleaning:
    def __init__(self, raw_data_path: str, cleaned_data_path: str):
        try:
            self.raw_data_path = raw_data_path
            self.cleaned_data_path = cleaned_data_path
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("Initialized DataCleaning with raw_data_path: %s and cleaned_data_path: %s", 
                         raw_data_path, cleaned_data_path)
        except Exception as e:
            logging.error("Error initializing DataCleaning: %s", str(e))
            raise CustomException(e)

    def read_data(self) -> pd.DataFrame:
        try:
            logging.info("Reading raw data from %s", self.raw_data_path)
            df = pd.read_csv(self.raw_data_path)
            logging.info("Successfully read data with shape: %s", df.shape)
            return df
        except Exception as e:
            logging.error("Error reading data: %s", str(e))
            raise CustomException(e)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling missing values...")
            missing_before = df.isnull().sum().sum()
            for column, dtype in self._schema_config.get("columns", {}).items():
                if column in df.columns:
                    if dtype in ["float64", "int64"]:
                        df[column] = df[column].fillna(df[column].median())
                    else:
                        df[column] = df[column].fillna(df[column].mode()[0])
            missing_after = df.isnull().sum().sum()
            logging.info("Missing values before: %d, after: %d", missing_before, missing_after)
            return df
        except Exception as e:
            logging.error("Error handling missing values: %s", str(e))
            raise CustomException(e)
        
    def handle_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling duplicates...")
            duplicates_before = df.duplicated().sum()
            df.drop_duplicates(inplace=True)
            duplicates_after = df.duplicated().sum()
            logging.info("Duplicates before: %d, after: %d", duplicates_before, duplicates_after)
            return df
        except Exception as e:
            logging.error("Error handling duplicates: %s", str(e))
            raise CustomException(e)
        
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Handling outliers...")
            for column in df.select_dtypes(include=[np.number]).columns:
                q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
                iqr = q3 - q1
                lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers_before = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
                df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
                outliers_after = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                logging.info("Column: %s, Outliers before: %d, after: %d", column, outliers_before, outliers_after)
            return df
        except Exception as e:
            logging.error("Error handling outliers: %s", str(e))
            raise CustomException(e)


    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        expected_dtypes = self._schema_config.get("columns", {})
        try:
            for column, expected_dtype in expected_dtypes.items():                
                if column in df.columns:
                    actual_dtype = str(df[column].dtype)

                    # Handle object vs string efficiently
                    if expected_dtype == "string" and actual_dtype == "object":
                        continue  
                    print(expected_dtype, actual_dtype)
                    if actual_dtype != expected_dtype:
                        try:
                            df[column] = self._convert_column(df[column], expected_dtype)
                            logging.info(f"Converted Column: {column} | From: {actual_dtype} → To: {expected_dtype}")
                        except Exception as e:
                            logging.error(f"Failed to convert column {column}: {e}")
            return df
        except Exception as e:
            logging.error("Error handling outliers: %s", str(e))
            raise CustomException(e)

    @staticmethod
    def _convert_column(series: pd.Series, expected_dtype: str) -> pd.Series:
        try:
            if "int" in expected_dtype:
                return pd.to_numeric(series, errors="coerce").astype("Int64")
            elif "float" in expected_dtype:
                return pd.to_numeric(series, errors="coerce").astype("float64")
            elif "datetime" in expected_dtype:
                if series.dtype in ["int64", "float64"]:
                    # Detect timestamp format based on value magnitude
                    max_ts = series.max()
                    if max_ts > 10**12:  # Likely in milliseconds
                        return pd.to_datetime(series, unit="ms", errors="coerce")
                    elif max_ts > 10**9:  # Likely in seconds
                        return pd.to_datetime(series, unit="s", errors="coerce")
                    else:  # If values are too small, assume normal datetime conversion
                        return pd.to_datetime(series, errors="coerce")
                return pd.to_datetime(series, errors="coerce")  # Handle string/object datetime
            elif expected_dtype == "string":
                return series.astype("string")
            else:
                logging.warning(f"Unknown dtype {expected_dtype}, keeping original dtype.")
                return series
        except Exception as e:
            logging.error(f"Error converting dtype: {e}")
            return series  # Keep original if conversion failss
