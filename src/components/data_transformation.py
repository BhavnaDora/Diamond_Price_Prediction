from sklearn.impute import SimpleImputer              # to handle missing values
from sklearn.preprocessing import StandardScaler      # standardizatiion
from sklearn.preprocessing import OrdinalEncoder      # ordinal encoding of categorical features
#pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import os, sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path= os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info("Data transformation starts")

            # Numerical and Categorical features-
            numerical_cols= ['carat', 'depth', 'table', 'x', 'y', 'z']
            categorical_cols= ['cut', 'color', 'clarity']

            # define custom ranking for categorical variables-
            cut_categories= ["Fair", "Good", "Very Good", "Premium", "Ideal"]
            color_categories= ["D", "E", "F", "G", "H", "I", "J"]
            clarity_categories= ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

            logging.info("creating pipelines to handle missing values, scaling, ordinal encoding")

            num_pipeline= Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline= Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy= "most_frequent")),
                    ("encoder", OrdinalEncoder(categories= [cut_categories, color_categories, clarity_categories])),
                    ("scaler", StandardScaler())
                ]
            )
            preprocessor= ColumnTransformer(
                [
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            logging.info("Data transformation completed")
            return preprocessor


        except Exception as e:
            logging.info("Exception occured at Data Transformation stage")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df= pd.read_csv(train_data_path)
            test_df= pd.read_csv(test_data_path)

            logging.info("Train and Test data are read")
            logging.info(f"Train data head: \n{train_df.head().to_string()}")
            logging.info(f"Test data head: \n{test_df.head().to_string()}")

            logging.info("Calling get_data_transformation_object() function to get preprocessor object")
            preprocessor= self.get_data_transformation_object()

            target_column= "price"
            drop_columns= ["id", target_column]

            logging.info("X_train, y_train, X_test, y_test-")
            input_feature_train_df= train_df.drop(columns= drop_columns, axis=1)
            target_feature_train_df= train_df[target_column]

            input_feature_test_df= test_df.drop(columns= drop_columns, axis=1)
            target_feature_test_df= test_df[target_column]

            # Data Transformation-
            logging.info("Applying preprocessor on X_train and X_test")
            input_feature_train_arr= preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr= preprocessor.transform(input_feature_test_df)

            train_arr= np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessor
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)
