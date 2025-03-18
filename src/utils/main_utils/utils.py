import yaml
from src.exception.exception import CustomException
from src.logging.logger import logging
import os,sys
import numpy as np
import pandas as pd
import json
#import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def save_numpy_array_data(file_path: str, df: pd.DataFrame):
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # Save as .npy
    np.save(file_path, df.to_numpy(), allow_pickle=True)

    # Save as CSV
    csv_path = file_path.replace(".npy", ".csv")
    df.to_csv(csv_path, index=False)

    # Save dtypes separately as .pkl
    dtypes_path = file_path.replace(".npy", "_dtypes.pkl")
    df.dtypes.to_pickle(dtypes_path)

    # Save dtypes in a human-readable format (JSON)
    dtypes_json_path = file_path.replace(".npy", "_dtypes.json")
    with open(dtypes_json_path, "w") as f:
        json.dump(df.dtypes.astype(str).to_dict(), f, indent=4)

    print(f"Data saved:\n- NPY: {file_path}\n- CSV: {csv_path}\n- Dtypes PKL: {dtypes_path}\n- Dtypes JSON: {dtypes_json_path}")
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of MainUtils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path: str ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> pd.DataFrame:
    try:
        # Load the data
        array = np.load(file_path, allow_pickle=True)

        # Load the dtypes
        dtypes_path = file_path.replace(".npy", "_dtypes.pkl")
        dtypes = pd.read_pickle(dtypes_path)
        print("dtypes -->",dtypes)
        # Convert back to DataFrame with correct dtypes
        df = pd.DataFrame(array, columns=dtypes.index)
        df = df.astype(dtypes.to_dict())  # Restore dtypes
        return df

    except Exception as e:
        raise CustomException(e, sys) from e

    


def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys) 