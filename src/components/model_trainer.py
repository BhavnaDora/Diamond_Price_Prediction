import os, sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path= os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("splitting the train_arr and test_arr into X_train, y_train, X_test, y_test")
            X_train, y_train, X_test, y_test= (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])

            models={
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "Elastic Net": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "K Neighbors": KNeighborsRegressor()
            }

            report:dict= evaluate_model(X_train, y_train, X_test, y_test, models)
            print("\n=================================================================================\n")
            logging.info(f"Model report: {report}")

            #best model score from report dictionary-
            best_model_score= max(report.values())

            #best model coreesponding to best model score-
            for i in list(report.keys()):
                if report[i]==best_model_score:
                    best_model_name= i

            best_model= models[best_model_name]
            
            print(f"Best model found- Model name: {best_model_name} and R2 score: {best_model_score}")
            print(f"\n=================================================================================\n")
            logging.info(f"Best model found- Model name: {best_model_name} and R2 score: {best_model_score}")

            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            logging.info("Model pickle file created")
        

        except Exception as e:
            logging.info("Exception occured at model training stage")
            raise CustomException(e,sys)