from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.components.data_cleaning import DataCleaning
from src.exception.exception import CustomException 
from src.logging.logger import logging 
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.data_cleaning = DataCleaning(
                raw_data_path=self.data_ingestion_artifact.feature_store_path,
                cleaned_data_path=self.data_validation_config.valid_data_dir
            )
        except Exception as e:
            raise CustomException(e, sys)
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_expected_columns(config):
        return list(config["columns"].keys())
    

    def validate_dtypes(self, dataframe: pd.DataFrame) -> bool:
        try:            
            expected_dtypes = self._schema_config.get("columns", {})
            validation_passed = True

            for column, expected_dtype in expected_dtypes.items():
                if column not in dataframe.columns:
                    logging.error(f"Missing column: {column} (Expected dtype: {expected_dtype})")
                    validation_passed = False
                    continue

                actual_dtype = str(dataframe[column].dtype)

                if expected_dtype == "string" and actual_dtype == "object":
                    continue

                if actual_dtype != expected_dtype:
                    logging.warning(f"Column {column} has incorrect dtype: Expected {expected_dtype}, Found {actual_dtype}")
                    validation_passed = False  

            return validation_passed

        except Exception as e:
            logging.error("Error in data type validation: %s", str(e))
            raise CustomException(e, sys)

    
    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = list(self._schema_config["columns"].keys()) 
            print(expected_columns , dataframe.columns)
            return set(dataframe.columns) == set(expected_columns)
        except Exception as e:
            raise CustomException(e, sys)
    
    def check_missing_values(self, dataframe: pd.DataFrame) -> bool:
        try:
            return not dataframe.isnull().sum().any()
        except Exception as e:
            raise CustomException(e, sys)
    
    def check_duplicate_rows(self, dataframe: pd.DataFrame) -> bool:
        try:
            return dataframe.duplicated().sum() == 0
        except Exception as e:
            raise CustomException(e, sys)
    
    def check_outliers(self, dataframe: pd.DataFrame, threshold=1.5) -> bool:
        try:
            status = True
            for column in dataframe.select_dtypes(include=['int64', 'float64']).columns:
                q1 = dataframe[column].quantile(0.25)
                q3 = dataframe[column].quantile(0.75)
                iqr = q3 - q1
                outliers = ((dataframe[column] < (q1 - threshold * iqr)) | (dataframe[column] > (q3 + threshold * iqr))).sum()
                if outliers > 0:
                    status = False
            return status
        except Exception as e:
            raise CustomException(e, sys)
    
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                d1, d2 = base_df[column], current_df[column]
                p_value = ks_2samp(d1, d2).pvalue
                drift_detected = p_value < threshold
                report[column] = {"p_value": float(p_value), "drift_status": drift_detected}
                if drift_detected:
                    status = False
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)
            return status
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            data_file_path = self.data_ingestion_artifact.feature_store_path
            dataframe = self.read_data(data_file_path)

            if not self.validate_dtypes(dataframe):
                logging.info("Data type mismatch detected, initiating data cleaning...")
                dataframe = self.data_cleaning.convert_data_types(dataframe) 

            if not self.validate_number_of_columns(dataframe):
                logging.error("Dataset does not contain the required columns.")
                return None

            if not self.check_missing_values(dataframe):
                logging.info("Missing values detected, initiating data cleaning...")
                dataframe = self.data_cleaning.handle_missing_values(dataframe)
            
            if not self.check_duplicate_rows(dataframe):
                logging.info("Duplicate rows detected, initiating data cleaning...")
                dataframe = self.data_cleaning.handle_duplicate_rows(dataframe)

            if not self.check_outliers(dataframe):
                logging.info("Outliers detected, initiating data cleaning...")
                dataframe = self.data_cleaning.handle_outliers(dataframe)

            valid_data_path = self.data_validation_config.valid_file_path
            os.makedirs(os.path.dirname(valid_data_path), exist_ok=True)
            dataframe.to_csv(valid_data_path, index=False, header=True)

            validation_artifact = DataValidationArtifact(
                valid_data_file_path=valid_data_path,
            )
            return validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
