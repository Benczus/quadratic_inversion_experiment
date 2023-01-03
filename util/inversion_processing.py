import os
import pickle
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from quadratic_polynomial import QuadraticPolynomial


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


def process_inversion_results(model=None, X_test=None, Y_test=None, ga_inv_value=None):
    quadratic = QuadraticPolynomial(1, 1, 2)
    if not any((model, X_test, Y_test, ga_inv_value)):
        with open("inversion_save", "rb+") as f:
            model, X_test, Y_test, ga_inv_value = pickle.load(f)
        diffs_true_prediction = []
        diffs_model_prediction = []
        best_inversions = []
        best_inversion_predictions = []
        best_inversions_calculations = []
        diffs_model_calculation = []
        diffs_true_calculation = []
        for x, y, g in zip(X_test, Y_test, ga_inv_value):
            best_inversions.append(g[0][0])
            best_inversion_predictions.append(
                best_inversion_prediction := model.predict(g[0])
            )
            best_inversions_calculations.append(quadratic.calculate(g[0][0]))
            diffs_true_prediction.append(y - best_inversion_prediction)
            diffs_true_calculation.append(y - quadratic.calculate(g[0][0]))
            diffs_model_prediction.append(
                model.predict(x.reshape(-1, 1)) - best_inversion_prediction
            )
            diffs_model_calculation.append(
                model.predict(x.reshape(-1, 1)) - quadratic.calculate(g[0][0])
            )
        results_dict = {
            "desired_values": [y[0] for y in Y_test],
            "inverted_input": [x[0] for x in X_test],
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


def save_ecdf_plots(results_df: pd.DataFrame, path: str = f""):
    for column in results_df.columns:
        sns.ecdfplot(results_df, x=column)
        plt.savefig(str(path) + f"/ecdf_plot_{column}")
        plt.close()
