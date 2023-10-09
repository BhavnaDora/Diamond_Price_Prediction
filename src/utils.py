import os, sys
import pandas as pd
import numpy as np
import pickle

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report= {}
        for i in range(len(models)):
            model= list(models.values())[i]

            #training the model-
            model.fit(X_train, y_train)

            #prediction-
            y_test_pred= model.predict(X_test)

            # evaluation-
            score= r2_score(y_test_pred, y_test)
            report[list(models.keys())[i]]= score

            return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e,sys)
