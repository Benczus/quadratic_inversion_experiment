import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ann_training import model_creation_2D, model_creation_3D
from experiment_utils.general import (inversion_3D, pipeline_ga_MLP_3D,
                                      pipeline_MLP_2D)
from experiment_utils.util import sort_2D_test_data
from function.Polynomial import Polynomial
from inversion_util import invert_mlp_ga_2_d


def main_ga_2_d(function: Polynomial, num_of_rows=100):
    if not os.path.exists(f"mlpmodel2D_{num_of_rows}"):
        function_return, model, x_test, y_test = pipeline_MLP_2D(
            function, num_of_rows=num_of_rows
        )
        pickle.dump(
            (function_return, model, x_test, y_test),
            open("mlpmodel2D_{num_of_rows}", "wb"),
        )
    else:
        function_return, model, x_test, y_test = pickle.load(
            open(f"mlpmodel2D_{num_of_rows}", "rb")
        )
    if not os.path.exists(f"inversion_save_{num_of_rows}"):
        bounds = (np.array(x_test.min(axis=1)), np.array(x_test.max(axis=1)))
        ga_inv_value = inversion_ga_2_d(bounds, model, y_test=y_test)
        print(ga_inv_value, np.array(x_test[0]))
        with open(f"inversion_save_{num_of_rows}", "wb+") as f:
            pickle.dump((model, x_test, y_test, ga_inv_value), f)
    else:
        with open(f"inversion_save_{num_of_rows}", "rb") as f:
            model, x_test, y_test, ga_inv_value = pickle.load(f)
    return model, x_test, y_test, ga_inv_value


def main_ga_3_d(function: Polynomial, num_of_rows=100):
    if not os.path.exists(f"mlpmodel3D_{num_of_rows}"):

        quadratic, model, quad_x_test, quad_y_test, quad_z_test = pipeline_ga_MLP_3D(
            function=function, num_of_rows=num_of_rows
        )
        pickle.dump(
            (quadratic, model, quad_x_test, quad_y_test, quad_z_test),
            open(f"mlpmodel3D_{num_of_rows}", "wb"),
        )
    else:
        quadratic, model, quad_x_test, quad_y_test, quad_z_test = pickle.load(
            open(f"mlpmodel3D_{num_of_rows}", "rb")
        )
    if not os.path.exists(f"inversion_save3D_{num_of_rows}"):
        bounds = (np.array(quad_x_test.min(axis=1)), np.array(quad_x_test.max(axis=1)))
        ga_inv_value = inversion_3D(bounds, model, quad_z_test, method="ga")
        with open(f"inversion_save3D_{num_of_rows}", "wb+") as f:
            pickle.dump((model, quad_x_test, quad_y_test, quad_z_test, ga_inv_value), f)
    else:
        with open(f"inversion_save3D_{num_of_rows}", "rb") as f:
            model, quad_x_test, quad_y_test, quad_z_test, ga_inv_value = pickle.load(f)
    # plot_3D(quadratic, model, quad_X_test, quad_Y_test, quad_Z_test)
    return model, quad_x_test, quad_y_test, quad_z_test, ga_inv_value


def inversion_ga_2_d(bounds, model, y_test):
    ga_inv_value = []
    for value in y_test:
        inveted_values = invert_mlp_ga_2_d(value=value, regressor=model, bounds=bounds)
        ga_inv_value.append(inveted_values[:10])
    return ga_inv_value


def invert_ga_2_d(bounds, model, y_test):
    ga_inv_value = invert_mlp_ga_2_d(y_test, model, bounds)
    # pickle.dump(
    #     ga_inv_value, open(f"ga_inv_value_2D_{len(ga_inv_value)}", "wb")
    # )
    return ga_inv_value


def pipeline_ga_2_d():
    num_of_rows = 2000
    df = pd.read_csv(f"data/quadratic_{num_of_rows}", index_col=0)
    # df = scale_dataset(df)
    X = df[["x", "y"]]
    y = df["z"]
    neuron_config = [200, 400, 300, 200]
    activation_config = ["selu", "tanh", "sigmoid", "exponential"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    X_test, y_test = sort_2D_test_data(X_test, y_test)
    model, new_loss = model_creation_2D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
    )
    return X_test, model, y_test


def pipeline_ga_3_d():
    quadratic = Polynomial([1, 1, 2])
    num_of_rows = 200
    quad_x, quad_y, quad_z = quadratic.generate_quadratic_data_3D(
        num_of_rows=num_of_rows
    )
    quadratic.plot_surface()
    x_train = np.append(quad_x, quad_y, axis=1)
    y_train = quad_z
    quad_x_test, quad_y_test, quad_z_test = quadratic.generate_quadratic_data_3D(
        num_of_rows=num_of_rows
    )
    # input can be only 1 dimensional. 2 sets of inputs -> add new columns
    neuron_config = [400, 600, 600, 600, 400]
    activation_config = ["selu", "selu", "sigmoid" "exponential", "linear"]
    model = model_creation_3D(X_train=x_train, activation_config=activation_config, neuron_config=neuron_config,
                              y_train=y_train, X_test=quad_x_test, y_test=quad_z_test)
    print("Done Training")
    return quadratic, model, quad_x_test, quad_y_test, quad_z_test
