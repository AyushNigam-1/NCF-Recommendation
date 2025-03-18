import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from src.exception.exception import CustomException  # Assuming you have this
from src.logging.logger import logging  # Assuming you have this
from src.utils.main_utils.utils import save_numpy_array_data, save_object
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)
from src.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact: DataValidationArtifact = data_validation_artifact
            self.data_transformation_config: DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    def get_data_transformer_object(self, content_series):
        try:
            logging.info("Creating data transformer object...")
            mlb = MultiLabelBinarizer()
            content_list = content_series.apply(lambda x: x.split(','))  # Ensure it's a list of lists
            mlb.fit(content_list)
            logging.info("Data transformer object created successfully.")
            return mlb
        except Exception as e:
            raise CustomException(e, sys)
        
    def perform_feature_engineering(self, combined_df:pd.DataFrame):
        try:
            combined_df['tag'] = combined_df['tag'].astype(str).str.strip()

            combined_df['tag'] = combined_df['tag'].fillna('')
            combined_df = combined_df.sort_values(by=['userId', 'movieId', 'timestamp'], ascending=[True, True, False])
            combined_df = combined_df.drop_duplicates(subset=['userId', 'movieId'], keep='first').reset_index(drop=True)
            combined_df["tag"] = combined_df["tag"].str.lower().str.replace(" ", ",")
            combined_df['genres'] = combined_df['genres'].str.lower().str.replace("|", ",")
            combined_df['content'] = combined_df['genres'] + ',' + combined_df['tag']
            combined_df['release_year'] = combined_df['title'].str.extract(r'\((\d{4})\)').astype(float)
            median_year = combined_df['release_year'].median()
            combined_df['release_year'] = combined_df['release_year'].fillna(median_year)
            current_time = combined_df['timestamp'].max() # Use timestamp
            combined_df['time_decay_weight'] = combined_df['timestamp'].apply(lambda x: np.exp(-0.001 * (current_time - x) / (60 * 60 * 24))) # Use timestamp
            current_year = combined_df['release_year'].max()
            combined_df['year_decay_weight'] = combined_df['release_year'].apply(lambda x: np.exp(-0.001 * (current_year - x)))
            combined_df['combined_decay'] = combined_df['time_decay_weight'] * combined_df['year_decay_weight']
            combined_df['userId'] -= 2
            combined_df['movieId'] -= 2
            return combined_df
        except Exception as e:
            raise CustomException(e, sys)
        
    def split_train_test(self, X: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        logging.info("Splitting dataset into train and test sets...")
        train , test= train_test_split(X, test_size=test_size, random_state=random_state)
        
        logging.info(f"Train shape: {train.shape}, Test shape: {test.shape}")
        
        return train, test 
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            logging.info("Starting data transformation")
            file_paths = self.data_validation_artifact.valid_data_file_path

            if len(file_paths) != 4:
                raise ValueError("Expected 4 file paths: movies, ratings, tags, links.")

            movies = self.read_data(file_paths[0])  # movies.csv
            ratings = self.read_data(file_paths[2])  # ratings.csv
            tags = self.read_data(file_paths[3])    # tags.csv
            #links = self.read_data(file_paths[1])    #links.csv (not used)

            merged_df = pd.merge(movies, tags, on='movieId')
            movies = merged_df.groupby(['movieId', 'title', 'genres'])['tag'].apply(lambda x: ', '.join(x.dropna().astype(str))).reset_index()
            combined_df = pd.merge(movies,ratings,on='movieId')
            combined_df = combined_df.sort_values(by=['userId', 'movieId', 'timestamp'], ascending=[True, True, False])
            df = self.perform_feature_engineering(combined_df)
            transformer = self.get_data_transformer_object(df['content'])
            genres_encoded = pd.DataFrame(transformer.transform(df['content'].apply(lambda x: x.split(','))), 
                                        columns=transformer.classes_, 
                                        index=df.index)
            df = pd.concat([df, genres_encoded], axis=1)
            
            train_df, test_df = self.split_train_test(df)
            # print(train_df.dtypes)
            # print(train_df.dtypes)
            
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, df=train_df)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, df=test_df)
            save_object(self.data_transformation_config.transformed_object_file_path, transformer)
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                dataframe_columns=df.columns
            )
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)
