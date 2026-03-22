import sys
import os
from dataclasses import dataclass

from seaborn import categorical
from sklearn import preprocessing

from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = get_logger(__name__)

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        log.info("Entered the Data Transformer Component (get_data_transformer)")
        '''
        This function transforms the data
        returns the preprocessor
        '''
        try:
            # Below columns as per EDA
            numerical_cols = [
                'reading score',
                'writing score'
            ] # removing the target col!!!!
            categorical_cols = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            # defining pipelines:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")), #Cuz of outliers
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")), #basically mode
                    ("one_hot_encoder",OneHotEncoder()), #text to num
                    ("scaler",StandardScaler(with_mean=False)) # just for the love of game <3
                    # alright this love of the game gave me a sparse matrix cry and now
                    # gotta work with centered data with variance without mean subtraction
                ]
            )
            # log here nigga
            log.info(
                f"established pipelines with cols as:\nnumerical columns:[{numerical_cols}]\ncategorical columns:[{categorical_cols}]"
            )
            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_cols),
                ("cat_pipeline",cat_pipeline, categorical_cols)
            ])
            log.info("established preprocessor, returning said object")

            return preprocessor
        except Exception as e:
            log.error(f"Error occurred: {e}", exc_info=True)
            raise CustomException(e,sys)

    def initiate_data_transformer(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            log.info("Successfully read train and test file")
            log.info("Obtaining preprocessing")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column_name = 'math score'

            input_feature_train_df = train_df.drop(
                columns=[target_column_name],axis=1
            )
            target_feature_train_df = train_df[target_column_name]


            input_feature_test_df = test_df.drop(
                columns=[target_column_name],axis=1
            )
            target_feature_test_df = test_df[target_column_name]

            log.info(
                f"Applying preprocessing object in training and testing dataframe."
            )
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )
            log.info(f'Saved preprocessing object')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            log.error(f"Error occurred: {e}", exc_info=True)
            raise CustomException(e,sys)