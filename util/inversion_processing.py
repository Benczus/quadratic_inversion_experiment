import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def save_results_as_excel(df, filename: str = "inversion_results.xlsx"):
    # path = Path(__file__) / "pure_scraping_results"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    with pd.ExcelWriter(
            str(filename),
    ) as writer:
        df.to_excel(writer)


def save_results_as_csv(df, filename: str = "inversion_results.csv"):
    with open(filename, "wb+") as file:
        df.to_csv(file)


def process_inversion_results_2_d(
        model=None, x_test=None, y_test=None, ga_inv_value=None, function=None
):
    if not any((model, x_test, y_test, ga_inv_value)):
        with open("inversion_save", "rb+") as f:
            model, x_test, y_test, ga_inv_value = pickle.load(f)
        diffs_true_prediction = []
        diffs_model_prediction = []
        best_inversions = []
        best_inversion_predictions = []
        best_inversions_calculations = []
        diffs_model_calculation = []
        diffs_true_calculation = []
        for x, y, g in zip(x_test, y_test, ga_inv_value):
            best_inversions.append(g[0][0])
            best_inversion_predictions.append(
                best_inversion_prediction := model.predict(g[0])
            )
            best_inversions_calculations.append(function.calculate(g[0][0]))
            diffs_true_prediction.append(y - best_inversion_prediction)
            diffs_true_calculation.append(y - function.calculate(g[0][0]))
            diffs_model_prediction.append(
                model.predict(x.reshape(-1, 1)) - best_inversion_prediction
            )
            diffs_model_calculation.append(
                model.predict(x.reshape(-1, 1)) - function.calculate(g[0][0])
            )
        results_dict = {
            "desired_values": [y[0] for y in y_test],
            "inverted_input": [x[0] for x in x_test],
            "inverted_output": [a[0] for a in best_inversions],
            "model_prediction_inversion": [a[0] for a in best_inversion_predictions],
            "calculated_inverted_value": [a[0] for a in best_inversions_calculations],
            "difference_desired_predicted": [a[0] for a in diffs_true_prediction],
            "difference_model_predicted": [a[0] for a in diffs_model_prediction],
            "difference_desired_calculated": [a[0] for a in diffs_true_calculation],
            "difference_model_calculation": [a[0] for a in diffs_model_calculation],
        }
        results_df = pd.DataFrame(results_dict)
        normalized_df = (results_df - results_df.min()) / (
                results_df.max() - results_df.min()
        )
        plots_path = Path(__file__).parent / "data" / "plots"
        if not os.path.exists(plots_path):
            os.makedirs(plots_path)
        save_ecdf_plots(normalized_df, path=str(plots_path))
        save_results_as_csv(results_df)
        save_results_as_excel(results_df)


def process_inversion_results_3_d(
        model=None, x_test=None, y_test=None, z_test=None, ga_inv_value=None, function=None, num_of_rows=None
):
    if not any((model, x_test, y_test, z_test, ga_inv_value)):
        with open(f"inversion_save3D_{num_of_rows}", "rb+") as f:
            model, x_test, y_test, z_test, ga_inv_value = pickle.load(f)
    diffs_true_prediction = []
    diffs_model_prediction = []
    best_inversions = []
    best_inversion_predictions = []
    best_inversions_calculations = []
    diffs_model_calculation = []
    diffs_true_calculation = []
    for x, y, z, g in zip(x_test, y_test, z_test, ga_inv_value):
        # TODO - itt kellene a z-t is hozzáadni

        best_inversions.append(g[0][0])
        best_inversion_predictions.append(
            best_inversion_prediction := model.predict(g)
        )
        best_inversions_calculations.append(function.calculate(g[0][0]))
        diffs_true_prediction.append(y - best_inversion_prediction)
        diffs_true_calculation.append(y - function.calculate(g[0][0]))
        diffs_model_prediction.append(
            model.predict(np.array([x[0], y[0]]).reshape(1, -1)) - best_inversion_prediction
        )
        diffs_model_calculation.append(
            model.predict(np.array([x[0], y[0]]).reshape(1, -1)) - function.calculate(g[0][0])
        )
    results_dict = {
        "desired_values": [z[0] for z in z_test],
        "inverted_input": [x[0] for x in x_test],
        "inverted_output": [a for a in best_inversions],
        "model_prediction_inversion": [a for a in best_inversion_predictions],
        "calculated_inverted_value": [a for a in best_inversions_calculations],
        "difference_desired_predicted": [a for a in diffs_true_prediction],
        "difference_model_predicted": [a for a in diffs_model_prediction],
        "difference_desired_calculated": [a for a in diffs_true_calculation],
        "difference_model_calculation": [a for a in diffs_model_calculation],
    }
    results_df = pd.DataFrame(results_dict)
    normalized_df = (results_df - results_df.min()) / (
            results_df.max() - results_df.min()
    )
    plots_path = Path(__file__).parent / "data" / "plots"
    if not os.path.exists(plots_path):
        os.makedirs(plots_path)
    save_ecdf_plots(normalized_df, path=str(plots_path))
    save_results_as_csv(results_df)
    save_results_as_excel(results_df)


def save_ecdf_plots(results_df: pd.DataFrame, path: str = f""):
    for column in results_df.columns:
        sns.ecdfplot(results_df, x=column)
        plt.savefig(str(path) + f"/ecdf_plot_{column}")
        plt.close()
