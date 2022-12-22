import pickle

import numpy as np
import pandas as pd

from quadratic_polynomial import QuadraticPolynomial


def save_results_as_excel(df, filename: str = "inversion_results.xlsx"):
    # path = Path(__file__) / "pure_scraping_results"
    # if not os.path.exists(path):
    #     os.makedirs(path)
    with pd.ExcelWriter(
        str(filename),
    ) as writer:
        df.to_excel(writer)


def process_inversion_results(model=None, X_test=None, Y_test=None, ga_inv_value=None):
    quadratic = QuadraticPolynomial(1, 1, 2)
    if not any((model, X_test, Y_test, ga_inv_value)):
        with open("inversion_save", "rb+") as f:
            model, X_test, Y_test, ga_inv_value = pickle.load(f)
        diffs_true = []
        diffs_model = []
        best_inversions = []
        best_inversion_predictions = []
        best_inversions_calculations = []
        for x, y, g in zip(X_test, Y_test, ga_inv_value):
            best_inversions.append(g[0][0])
            best_inversion_predictions.append(
                best_inversion_prediction := model.predict(g[0])
            )
            best_inversions_calculations.append(quadratic.calculate(g[0][0]))
            diffs_true.append(y - best_inversion_prediction)
            diffs_model.append(
                model.predict(x.reshape(-1, 1)) - best_inversion_prediction
            )
        print(np.mean(diffs_true))
        print(np.mean(diffs_model))
        results_dict = {
            "desired_values": [y[0] for y in Y_test],
            "inverted_input": [x[0] for x in X_test],
            "model_prediction_inversion": best_inversion_predictions,
            "calculated_inverted_value": best_inversions_calculations,
            "difference_desired_predicted": diffs_true,
            "difference_model_predicted": diffs_model,
        }
        save_results_as_excel(pd.DataFrame(results_dict))
