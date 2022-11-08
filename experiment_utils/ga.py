import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ann_training import model_creation_2D, model_creation_3D
from experiment_utils.general import (inversion_3D, pipeline_ga_MLP_3D,
                                      pipeline_MLP_2D)
from experiment_utils.util import sort_2D_test_data
from inversion_util import invert_MLP_GA_2D
from plotting import plot_3D, plot_inversion_2D
from quadratic_polynomial import QuadraticPolynomial


def main_ga_2D():
    p = QuadraticPolynomial(4, 1, 2)
    num_of_rows = 200
    # if not os.path.isfile(os.pardir + f"/data/quadratic_{num_of_rows}_2D.csv"):
    p.generate_quadratic_data_2D(num_of_rows)
    p.save_data_2D()
    # if not os.path.exists("mlpmodel2D"):
    X_test, model, y_test = pipeline_MLP_2D(num_of_rows)
    # pickle.dump((X_test, model, y_test), open("mlpmodel2D", "wb"))
    # else:
    #     X_test, model, y_test = pickle.load(open("mlpmodel2D", "rb"))
    bounds = (np.array([X_test["x"].min()]), np.array([X_test["x"].max()]))
    # if not os.path.exists("ga_inv_value_2D_100"):
    ga_inv_value = inversion_ga_2D(bounds, model, y_test)
    # else:
    #     ga_inv_value= pickle.load(open("ga_inv_value_2D_100", "rb"))

    print(ga_inv_value[0][0], np.array(X_test)[0])
    plot_inversion_2D(model, ga_inv_value[0], X_test, y_test)
    return model, X_test, ga_inv_value


def main_ga_3D():
    if not os.path.exists("mlpmodel3D"):
        quadratic, model, quad_X_test, quad_Y_test, quad_Z_test = pipeline_ga_MLP_3D()
        pickle.dump(
            (quadratic, model, quad_X_test, quad_Y_test, quad_Z_test),
            open("mlpmodel3D", "wb"),
        )
    else:
        quadratic, model, quad_X_test, quad_Y_test, quad_Z_test = pickle.load(
            open("mlpmodel3D", "rb")
        )
    bounds = (np.array(quad_X_test.min(axis=1)), np.array(quad_X_test.max(axis=1)))
    ga_inv_value = inversion_3D(bounds, model, quad_Z_test, method="ga")
    print(ga_inv_value[0], np.array(quad_X_test)[0])
    plot_3D(quadratic, model, quad_X_test, quad_Y_test, quad_Z_test)
    return model, quad_X_test, ga_inv_value


def inversion_ga_2D(bounds, model, y_test):
    ga_inv_value = invert_MLP_GA_2D(value=y_test, regressor=model, bounds=bounds)
    return ga_inv_value


def invert_ga_2D(bounds, model, y_test):
    ga_inv_value = invert_MLP_GA_2D(y_test, model, bounds)
    # pickle.dump(
    #     ga_inv_value, open(f"ga_inv_value_2D_{len(ga_inv_value)}", "wb")
    # )
    return ga_inv_value


def pipeline_ga_2D():
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


def pipeline_ga_3D():
    quadratic = QuadraticPolynomial(1, 1, 2)
    num_of_rows = 200
    quad_X, quad_Y, quad_Z = quadratic.generate_quadratic_data_3D(
        num_of_rows=num_of_rows
    )
    quadratic.plot_surface()
    X_train = np.append(quad_X, quad_Y, axis=1)
    y_train = quad_Z
    quad_X_test, quad_Y_test, quad_Z_test = quadratic.generate_quadratic_data_3D(
        num_of_rows=num_of_rows
    )
    # input can be only 1 dimensional. 2 sets of inputs -> add new columns
    neuron_config = [400, 600, 600, 600, 400]
    activation_config = ["selu", "selu", "sigmoid" "exponential", "linear"]
    model = model_creation_3D(X_train, activation_config, neuron_config, y_train)
    print("Done Training")
    return quadratic, model, quad_X_test, quad_Y_test, quad_Z_test
