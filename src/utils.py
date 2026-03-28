import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.model_selection import train_test_split, cross_val_score

from src.exception import CustomException
from src.logger import get_logger

import dill

log = get_logger(__name__)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        log.error(f"Error occurred: {e}", exc_info=True)
        raise CustomException(e,sys)


def evaluate_models(
    X_train, y_train,
    X_test, y_test,
    models: dict
) -> pd.DataFrame:
    """
    Train and evaluate multiple models.

    Returns:
        pd.DataFrame sorted by Test R2 score
    """

    results = []

    for name, model in models.items():
        print(f"Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        results.append({
            "Model": name,
            "Train R2": r2_score(y_train, y_train_pred),
            "Test R2": r2_score(y_test, y_test_pred),
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred)
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by best performing model
    results_df.sort_values(by="Test R2", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    return results_df

def get_regression_metrics(y_true, y_pred, model_name=None, verbose=True, plot=False):
    """
    Calculate standard regression metrics and optionally print and visualize them.

    Parameters
    ----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    model_name : str, optional
        Name of the model (for printing)
    verbose : bool
        Whether to print metrics
    plot : bool
        Whether to plot visualizations

    Returns
    -------
    metrics : dict
        Dictionary containing MAE, MSE, RMSE, R2
    """

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    if verbose:
        if model_name:
            print(f"--- {model_name} ---")
        print(f"MAE  : {mae:.2f}")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R2   : {r2:.4f}")
        print("-"*40)

    # Visualizations
    if plot:
        plt.figure(figsize=(16,5))

        # 1️⃣ True vs Predicted Scatter
        plt.subplot(1,2,1)
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted {'(' + model_name + ')' if model_name else ''}")

        # 2️⃣ Residual Plot
        plt.subplot(1,2,2)
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, bins=30, color='orange')
        plt.xlabel("Residuals")
        plt.title(f"Residuals Distribution {'(' + model_name + ')' if model_name else ''}")

        plt.tight_layout()
        plt.show()

    return metrics


def tune_models(
    X_train, y_train,
    X_test, y_test,
    models: dict,
    param_grids: dict,
    top_models: list,
    n_iter: int = 20
):
    tuned_results = {}
    best_models = {}

    for name in top_models:
        try:
            print(f"\n🔧 Tuning {name}...")

            model = models[name]
            params = param_grids.get(name)

            if params is None:
                print(f"⚠️ No param grid for {name}, skipping...")
                continue

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=n_iter,
                scoring="r2",
                cv=3,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            # Predictions
            y_pred = best_model.predict(X_test)

            # Metrics (your function)
            metrics = get_regression_metrics(
                y_test,
                y_pred,
                model_name=name,
                verbose=True,
                plot=False
            )

            tuned_results[name] = {
                "best_params": search.best_params_,
                "metrics": metrics
            }

            best_models[name] = best_model

        except Exception as e:
            print(f"❌ Error tuning {name}: {e}")
            continue

    return tuned_results, best_models