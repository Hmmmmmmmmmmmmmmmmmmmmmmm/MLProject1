import os
import sys
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.model_selection import RandomizedSearchCV

from src.exception import CustomException
from src.logger import get_logger

import dill

log = get_logger(__name__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def save_object(file_path, obj, type = "dill"):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        if type=="dill":
            with open(file_path, 'wb') as file_obj:
                dill.dump(obj, file_obj)
        elif type=="json":
             with open(file_path, "w") as f:
                json.dump(obj, f, indent=4,default=str)
        else:
            raise ValueError(f"Unsupported type: {type}")
    except Exception as e:
        log.error(f"Error occurred: {e}", exc_info=True)
        raise CustomException(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(
    X_train, y_train,
    X_test, y_test,
    models: dict,
    verbose = True,
    logs = True,
    save_results = True
) -> pd.DataFrame:
    """
    Train and evaluate multiple models.
    Returns: pd.DataFrame sorted by Test R2 score
    """

    results = []

    for name, model in models.items():
        if verbose:
            print(f"Training {name}...")
        if logs:
            log.info(f"Evaluating models: Training {name}...")
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

    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # report_dir = os.path.join("artifacts", "reports")
        report_dir = os.path.join(PROJECT_ROOT, "artifacts", "reports")
        os.makedirs(report_dir, exist_ok=True)
        csv_path = os.path.join(report_dir, f"baseline_model_scores_{timestamp}.csv")
        results_df.to_csv(csv_path, index=False)
        if logs:
            log.info(f"Saved baseline model scores to {csv_path}")
        plt.figure(figsize=(10,6))
        sns.barplot(data=results_df, x="Test R2", y="Model")
        plt.title("Baseline Model Comparison (R2 Score)")
        plt.tight_layout()
        plot_path = os.path.join(report_dir, f"baseline_model_comparison_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        if logs:
            log.info(f"Saved baseline comparison plot to {plot_path}")


    return results_df

def get_regression_metrics(y_true, y_pred, model_name=None, verbose=True, plot=False, logs = True):
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
    if logs:
        if model_name:
            log.info(f"Calculating Regression Metrics")
            log.info(f"--- {model_name} ---")
            log.info(f"MAE  : {mae:.2f}")
            log.info(f"MSE  : {mse:.2f}")
            log.info(f"RMSE : {rmse:.2f}")
            log.info(f"R2   : {r2:.4f}")
            log.info("-"*40)

    # Visualizations
    if plot:
        # plt.figure(figsize=(16,5))

        # # 1️⃣ True vs Predicted Scatter
        # plt.subplot(1,2,1)
        # sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        # plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
        # plt.xlabel("Actual Values")
        # plt.ylabel("Predicted Values")
        # plt.title(f"Actual vs Predicted {'(' + model_name + ')' if model_name else ''}")

        # # 2️⃣ Residual Plot
        # plt.subplot(1,2,2)
        # residuals = y_true - y_pred
        # sns.histplot(residuals, kde=True, bins=30, color='orange')
        # plt.xlabel("Residuals")
        # plt.title(f"Residuals Distribution {'(' + model_name + ')' if model_name else ''}")

        # plt.tight_layout()
        # plt.show()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # report_dir = os.path.join("artifacts", "reports", "regression_plots")
        report_dir = os.path.join(PROJECT_ROOT, "artifacts", "reports", "regression_plots")
        os.makedirs(report_dir, exist_ok=True)
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        # True vs Predicted
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted ({model_name})")
        # residuals
        plt.subplot(1,2,2)
        residuals = y_true - y_pred
        sns.histplot(residuals, kde=True, bins=30)
        plt.title(f"Residuals ({model_name})")
        plt.tight_layout()
        plot_path = os.path.join(
            report_dir,
            f"{model_name}_regression_{timestamp}.png"
        )
        plt.savefig(plot_path)
        plt.close()
        if logs:
                log.info(f"Saved regression plot for {model_name} at {plot_path}")
    return metrics


def tune_models(
    X_train, y_train,
    X_test, y_test,
    models: dict,
    param_grids: dict,
    top_models: list,
    n_iter: int = 100,
    verbose = True,
    logs = True
):
    tuned_results = {}
    best_models = {}
    log.info(f"Tuning Model (HyperParameter Tuning)")
    for name in top_models:
        try:
            if verbose:
                print(f"\n Tuning {name}...")
            if logs:
                log.info(f"Tuning {name}...")
            model = models[name]
            params = param_grids.get(name)

            if params is None:
                if verbose:
                    print(f"No param grid for {name}, skipping...")
                if logs:
                    log.info(f"No param grid for {name}, skipping...")
                continue

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=n_iter,
                scoring="r2",
                cv=5,
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
                plot=True
            )

            tuned_results[name] = {
                "best_params": search.best_params_,
                "metrics": metrics
            }

            best_models[name] = best_model

        except Exception as e:
            log.error(f"Error occurred: {e}", exc_info=True)
            if verbose:
                print(f" Error tuning {name}: {e}")

            continue
            # raise CustomException(e,sys)
    # report_dir = os.path.join("artifacts", "reports")
    report_dir = os.path.join(PROJECT_ROOT, "artifacts", "reports")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tuned_list = []
    for model, data in tuned_results.items():
        row = {
            "Model": model,
            "R2": data["metrics"]["R2"],
            "MAE": data["metrics"]["MAE"],
            "MSE": data["metrics"]["MSE"],
            "RMSE": data["metrics"]["RMSE"],
            "Best Params": str(data["best_params"])
        }
        tuned_list.append(row)

    tuned_df = pd.DataFrame(tuned_list)
    tuned_df.sort_values(by="R2", ascending=False, inplace=True)
    # Save CSV
    csv_path = os.path.join(report_dir, f"tuned_model_scores_{timestamp}.csv")
    tuned_df.to_csv(csv_path, index=False)

    if logs:
        log.info(f"Saved tuned model scores to {csv_path}")
    # Plot comparison
    plt.figure(figsize=(10,6))
    sns.barplot(data=tuned_df, x="R2", y="Model")
    plt.title("Tuned Model Comparison (R2 Score)")
    plt.tight_layout()
    plot_path = os.path.join(report_dir, f"tuned_model_comparison_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    if logs:
        log.info(f"Saved tuned comparison plot to {plot_path}")

    return tuned_results, best_models