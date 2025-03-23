from dataclasses import dataclass , field
from typing import Optional , List
import pandas as pd

@dataclass
class DataIngestionArtifact:
    feature_store_path:str
    files_name:List[str]


@dataclass
class DataValidationArtifact:
    valid_data_file_path: List[str]  # Required field (must be before default fields)
    drift_report_file_path: Optional[str] = field(default=None)
    validation_status: Optional[bool] = field(default=None)

@dataclass
class DataTransformationArtifact:
    transformed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    dataframe:pd.DataFrame


@dataclass
class RegressionMetricArtifact:
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    r2: float  # R² Score

    
@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
