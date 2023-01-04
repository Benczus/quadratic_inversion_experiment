import os
import pickle

import numpy as np

from experiment_utils.general import pipeline_MLP_2D
from inversion_util import invert_MLP_WLK_2D
from plotting import plot_inversion_2D
from polynomial import Polynomial


def inversion_wlk_2D(bounds, model, y_test):
    p = Polynomial([4, 1, 2])
    num_of_rows = 200
    wlk_inv_value = invert_wlk_2D(bounds, model, y_test)
    return wlk_inv_value


def main_wlk_2D():
    p = Polynomial([4, 1, 2])
    num_of_rows = 200
    if not os.path.isfile(os.pardir + f"/data/quadratic_{num_of_rows}.csv"):
        p.generate_quadratic_data_2D(num_of_rows)
        p.save_surface_2D()
    if not os.path.exists("mlpmodel2D"):
        X_test, model, y_test = pipeline_MLP_2D(num_of_rows)
        pickle.dump((X_test, model, y_test), open("mlpmodel2D", "wb"))
    else:
        X_test, model, y_test = pickle.load(open("mlpmodel2D", "rb"))
    bounds = (np.array(X_test.min(axis=1)), np.array(X_test.max(axis=1)))
    wlk_inv_value = inversion_wlk_2D(bounds, model, y_test)
    plot_inversion_2D(model, wlk_inv_value, X_test, y_test)
    return model, X_test, wlk_inv_value


def invert_wlk_2D(bounds, model, y_test):
    ga_inv_value, wlk_inv_value = invert_MLP_WLK_2D(bounds, model, y_test)
    return ga_inv_value, wlk_inv_value
