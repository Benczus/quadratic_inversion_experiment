import pickle

import numpy as np

from ann_training import model_creation_ga_MLP_3D, model_creation_MLP_2D
from function.Polynomial import Polynomial
from inversion_util import (invert_mlp_ga_2_d, invert_mlp_ga_3_d,
                            invert_mlp_wlk_2_d, invert_mlp_wlk_3_d)


def pipeline_ga_MLP_3D(function: Polynomial, num_of_rows: int = 100):
    quad_X, quad_Y, quad_Z = function.generate_quadratic_data_3D(
        num_of_rows=num_of_rows
    )
    # function.plot_surface()
    X_train = np.append(quad_X, quad_Y, axis=1)
    y_train = quad_Z
    quad_X_test, quad_Y_test, quad_Z_test = function.generate_quadratic_data_3D(
        num_of_rows=num_of_rows
    )
    # input can be only 1 dimensional. 2 sets of inputs -> add new columns
    X_test = np.append(quad_X_test, quad_Y_test, axis=1)
    y_test = quad_Z_test
    neuron_config = [400, 600, 600, 600, 400]
    activation_config = ["selu", "selu", "sigmoid" "exponential", "linear"]
    model, score = model_creation_ga_MLP_3D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
    )
    print("Done Training")
    return function, model, quad_X_test, quad_Y_test, quad_Z_test


def pipeline_MLP_2D(quadratic: Polynomial, num_of_rows: int = 100):
    X_train, y_train = quadratic.generate_quadratic_data_2D(num_of_rows=num_of_rows)
    # quadratic.plot_surface()
    X_test, y_test = quadratic.generate_quadratic_data_2D(num_of_rows=num_of_rows)
    # input can be only 1 dimensional. 2 sets of inputs -> add new columns
    neuron_config = [400, 600, 600, 600, 400]
    activation_config = ["selu", "selu", "sigmoid" "exponential", "linear"]
    model, score = model_creation_MLP_2D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
    )
    print("Done Training")
    return quadratic, model, X_test, y_test


def inversion_3D(bounds, model, y_test, method="ga", save=False, load=False):
    inv_value = None
    if not load:
        if method.lower() == "ga":
            inv_value = invert_mlp_ga_3_d(y_test[0], model, bounds=bounds)
        if method.lower() == "wlk":
            inv_value = invert_mlp_wlk_3_d(y_test[0], model, bounds=bounds)
        if save:
            pickle.dump((inv_value, y_test[0]), open(f"{method}_inv_value_2D", "wb"))
    else:
        inv_value = pickle.load(open("f{method}_inv_value_3D", "rb"))
    return inv_value


def inversion_2D(bounds, model, y_test, method="ga", save=False, load=False):
    inv_value = None
    if not load:
        if method.lower() == "ga":
            inv_value = invert_mlp_ga_2_d(y_test[0], model, bounds=bounds)
        if method.lower() == "wlk":
            inv_value = invert_mlp_wlk_2_d(y_test[0], model, bounds=bounds)
        if save:
            pickle.dump((inv_value, y_test[0]), open(f"{method}_inv_value_2D", "wb"))
    else:
        inv_value = pickle.load(open("f{method}_inv_value_3D", "rb"))
    return inv_value
