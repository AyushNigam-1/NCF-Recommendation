import os
import sys
from src.exception.exception import CustomException 
from src.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.ml_utils.model.estimator import MLModel
from src.utils.main_utils.utils import save_object,load_object
from src.utils.main_utils.utils import load_numpy_array_data,evaluate_models
from src.utils.ml_utils.metric.regression_metric import get_regression_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import os        
import mlflow
import dagshub

dagshub.init(repo_owner='ayushnigam843', repo_name='Dynamic-Pricing-Online-Retail', mlflow=True)


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)


    def track_mlflow(self, best_model, regression_metric):

        with mlflow.start_run():
            mlflow.log_metric("MAE", regression_metric.mae)
            mlflow.log_metric("MSE", regression_metric.mse)
            mlflow.log_metric("RMSE", regression_metric.rmse)
            mlflow.log_metric("R2 Score", regression_metric.r2)

            mlflow.sklearn.log_model(best_model, "model")


    def train_model(self, X_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestRegressor(verbose=1),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(verbose=1),
            "Linear Regression": LinearRegression(),
            "AdaBoost": AdaBoostRegressor(),
        }

        params = {
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [0.1, 0.01, 0.05, 0.001],
                'subsample': [0.6, 0.7, 0.75, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Linear Regression": {},  # No hyperparameters to tune
            "AdaBoost": {
                'learning_rate': [0.1, 0.01, 0.001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

        model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=x_test, y_test=y_test,
                                            models=models, param=params)

        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]

        y_train_pred = best_model.predict(X_train)
        train_metric = get_regression_score(y_true=y_train, y_pred=y_train_pred)

        self.track_mlflow(best_model, train_metric)

        y_test_pred = best_model.predict(x_test)
        test_metric = get_regression_score(y_true=y_test, y_pred=y_test_pred)
        self.track_mlflow(best_model, test_metric)

        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_Model = MLModel(preprocessor=preprocessor, model=best_model)
        save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model)
        save_object("final_model/model.pkl", best_model)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )

        return model_trainer_artifact

    
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise CustomException(e,sys)