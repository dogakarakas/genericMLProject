import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self): 
       self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transoformer_object(self):
        try:
            num_cols = ["writing score", "reading score"]
            cat_cols = ["gender",
                        "race/ethnicity",
                        "parental level of education",
                        "lunch",
                        "test preparation course"]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder())
                ]
            )
            logging.info("Numerical columns scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                [("num_pipeline", num_pipeline, num_cols),
                 ("cat_pipeline", cat_pipeline, cat_cols)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    
    def initate_data_transformation(self, train_path, test_path):
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transoformer_object()
            
            target_col_name = "math score"
            numerical_cols = ["reading score", "writing score"]
            
            input_feature_train_df = train_df.drop(columns=[target_col_name], axis=1)
            target_feature_train_df = train_df[target_col_name]
            
            input_feature_test_df = test_df.drop(columns=[target_col_name], axis=1)
            target_feature_test_df = test_df[target_col_name]
            
            logging.info("Applysing preprocessing obj on training and test dataframes")
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saved preprocessing obj")
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            raise CustomException(e,sys)
            