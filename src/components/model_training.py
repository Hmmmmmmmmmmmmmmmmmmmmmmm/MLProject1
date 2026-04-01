import os
import sys
from dataclasses import dataclass
from unittest.mock import patch
from datetime import datetime

# Models
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import get_logger
from src.exception import CustomException
from src.utils import (
    save_object,
    evaluate_models,
    tune_models
)


log = get_logger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@dataclass
class ModelTrainConfig:
    # trained_model_file_path:str = os.path.join("artifacts","model.pkl")
    trained_model_file_path:str = os.path.join(PROJECT_ROOT, "artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            log.info("split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            # now se select which model to use:
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "K-Neighbors Regression": KNeighborsRegressor(),
                "SVR": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                # "AdaBoost": AdaBoostRegressor(), As sklearn by default set estimator to None gotta specify this
                "AdaBoost": AdaBoostRegressor(estimator=DecisionTreeRegressor()),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0),
            }
            results_df  = evaluate_models(
                X_train = X_train,
                y_train=y_train,
                X_test=X_test,
                y_test = y_test,
                models = models
            )
            # parameters of models
            dt_params = {
                "max_depth": [None, 5, 10, 15, 20],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["squared_error", "friedman_mse", "absolute_error"]
            }
            rf_params = {
                "n_estimators": [100, 200, 500, 1000],
                "max_depth": [None, 5, 8, 10, 15],
                "max_features": ["sqrt", "log2", 5, 7, 8],
                "min_samples_split": [2, 8, 15, 20],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False]
            }
            ridge_params = {
                "alpha": [0.01, 0.1, 1, 10, 100],
                "solver": ["auto", "svd", "cholesky", "saga"]
            }
            lasso_params = {
                "alpha": [0.01, 0.1, 1, 10, 100],
                "max_iter": [1000, 5000, 10000],
                "selection": ["cyclic", "random"]
            }
            elastic_params = {
                "alpha": [0.01, 0.1, 1, 10, 100],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                "max_iter": [1000, 5000, 10000],
                "selection": ["cyclic", "random"]
            }
            gb_params = {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.6, 0.8, 1.0],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None]
            }
            ada_params = {
                "n_estimators": [100, 300, 500],
                "learning_rate": [0.01, 0.05, 0.1],
                "estimator__max_depth": [1, 2, 3, 4, 5],
                "estimator__min_samples_leaf": [1, 2, 4, 8]
            }
            xgb_params = {
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
            cat_params = {
                "iterations": [200, 500],
                "learning_rate": [0.01, 0.1],
                "depth": [4, 6, 8]
            }
            knn_params = {
                "n_neighbors": [3, 5, 7, 9, 11],
                "weights": ["uniform", "distance"],
                "p": [1, 2]
            }
            svr_params = {
                "C": [0.1, 1, 10, 100],
                "epsilon": [0.01, 0.1, 0.5],
                "kernel": ["rbf", "linear"]
            }
            param_grids = {
                "Decision Tree": dt_params,
                "Random Forest": rf_params,
                "Ridge": ridge_params,
                "Lasso": lasso_params,
                "ElasticNet": elastic_params,
                "K-Neighbors Regression": knn_params,
                "SVR": svr_params,
                "Gradient Boosting": gb_params,
                "AdaBoost": ada_params,
                "XGBoost": xgb_params,
                "CatBoost": cat_params
            }
            # Based on top 5 r2-score
            top_models = results_df.head(5)["Model"].tolist()
            tuned_results, best_models = tune_models(
                X_train, y_train,
                X_test, y_test,
                models,
                param_grids,
                top_models
            )


            best_model_name = max(
                tuned_results,
                key=lambda x: tuned_results[x]["metrics"]["R2"]
            )

            best_model = best_models[best_model_name]
            best_r2_score = tuned_results[best_model_name]["metrics"]["R2"]
            if best_r2_score<0.65:
                log.info(f"No best model found [r2_score<65]")
                raise CustomException("No best model found [r2_score<65]")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            summary = {
                "best_model": best_model_name,
                "r2_score": best_r2_score,
                "metrics": tuned_results[best_model_name]["metrics"],
                "best_params": tuned_results[best_model_name]["best_params"],
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            # summary_path = os.path.join("artifacts", "reports", f"best_model_summary_{summary['timestamp']}.json")
            summary_path = os.path.join(PROJECT_ROOT, "artifacts", "reports", f"best_model_summary_{summary['timestamp']}.json")
            save_object(
                file_path=summary_path,
                obj=summary,
                type="json"
            )

            log.info(f"Best model saved: {best_model_name}")
            log.info(f"Best R2 Score: {best_r2_score:.4f}")

            return summary

        except Exception as e:
            log.error(f"Error occurred: {e}", exc_info=True)
            raise CustomException(e,sys)



