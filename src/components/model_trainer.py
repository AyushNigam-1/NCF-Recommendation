import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from src.exception.exception import CustomException
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils.utils import save_object, load_object, load_numpy_array_data, evaluate_models
from src.utils.ml_utils.metric.regression_metric import get_regression_score

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def train_model(self, train, test, mlb):
        train['userId'], user_index = pd.factorize(train['userId'])
        train['movieId'], movie_index = pd.factorize(train['movieId'])

        test['userId'] = test['userId'].map(lambda x: user_index.get_loc(x) if x in user_index else -1)
        test['movieId'] = test['movieId'].map(lambda x: movie_index.get_loc(x) if x in movie_index else -1)
        test = test[(test['userId'] != -1) & (test['movieId'] != -1)]

        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')
        genre_input = Input(shape=(len(mlb.classes_),), name='genre_input')

        user_embedding = Embedding(input_dim=len(user_index), output_dim=50, name='user_embedding')(user_input)
        item_embedding = Embedding(input_dim=len(movie_index), output_dim=50, name='item_embedding')(item_input)

        user_vector = Flatten()(user_embedding)
        item_vector = Flatten()(item_embedding)

        concat = Concatenate()([user_vector, item_vector, genre_input])

        dense_1 = Dense(128, activation='relu')(concat)
        dense_1 = Dropout(0.3)(dense_1)
        dense_2 = Dense(64, activation='relu')(dense_1)
        output = Dense(1, activation='sigmoid')(dense_2)

        model = Model([user_input, item_input, genre_input], output)
        model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])
        model.summary()

        train_user = train['userId'].values
        train_item = train['movieId'].values
        train_rating = (train['rating'].values / 5.0) * train['combined_decay'].values
        train_genres = train[mlb.classes_].values

        test_user = test['userId'].values
        test_item = test['movieId'].values
        test_rating = (test['rating'].values / 5.0) * test['combined_decay'].values
        test_genres = test[mlb.classes_].values

        history = model.fit(
            [train_user, train_item, train_genres], train_rating,
            validation_data=([test_user, test_item, test_genres], test_rating),
            epochs=10, batch_size=256, verbose=1
        )

        train_predictions = model.predict([train_user, train_item, train_genres])
        test_predictions = model.predict([test_user, test_item, test_genres])

        train_metric = get_regression_score(y_true=train_rating, y_pred=train_predictions)
        test_metric = get_regression_score(y_true=test_rating, y_pred=test_predictions)

        save_object(self.model_trainer_config.trained_model_file_path, obj=model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )
        
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            mlb_file_path = self.data_transformation_artifact.transformed_object_file_path
            dataframe_columns = self.data_transformation_artifact.dataframe_columns

            train = load_numpy_array_data(train_file_path)
            test = load_numpy_array_data(test_file_path)

            train = pd.DataFrame(train, columns=dataframe_columns)
            test = pd.DataFrame(test, columns=dataframe_columns)

            mlb = load_object(mlb_file_path)
            # print(train.dtypes)
            # print(test.dtypes)

            model_trainer_artifact = self.train_model(train, test, mlb)
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)

        
        
        
    # def track_mlflow(self, best_model, regression_metric):

    #     with mlflow.start_run():
    #         mlflow.log_metric("MAE", regression_metric.mae)
    #         mlflow.log_metric("MSE", regression_metric.mse)
    #         mlflow.log_metric("RMSE", regression_metric.rmse)
    #         mlflow.log_metric("R2 Score", regression_metric.r2)

    #         mlflow.sklearn.log_model(best_model, "model")