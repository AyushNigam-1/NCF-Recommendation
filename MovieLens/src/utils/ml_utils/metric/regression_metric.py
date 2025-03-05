from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import sys
from src.exception.exception import CustomException
from src.entity.artifact_entity import RegressionMetricArtifact

def get_regression_score(y_true, y_pred) -> RegressionMetricArtifact:
    try:
        model_mae = mean_absolute_error(y_true, y_pred)
        model_mse = mean_squared_error(y_true, y_pred)
        model_rmse = np.sqrt(model_mse)
        model_r2 = r2_score(y_true, y_pred)

        regression_metric = RegressionMetricArtifact(
            mae=model_mae,
            mse=model_mse,
            rmse=model_rmse,
            r2=model_r2
        )
        return regression_metric
    except Exception as e:
        raise CustomException(e, sys)