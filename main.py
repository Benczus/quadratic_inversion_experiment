import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ann_training import (model_creation_2D, model_creation_3D,
                          model_creation_ga_MLP_3D, model_creation_MLP_2D)
from inversion_util import (invert_MLP_GA_2D, invert_MLP_GA_3D,
                            invert_MLP_WLK_2D, invert_MLP_WLK_3D)
from plotting import plot_2D, plot_3D, plot_inversion_2D, plot_inversion_3D
from quadratic_polynomial import QuadraticPolynomial

logger = logging.getLogger("exp_logger")


def inversion_wlk_2D(bounds, model, y_test):
    p = QuadraticPolynomial(4, 1, 2)
    num_of_rows = 200
    wlk_inv_value= invert_wlk_2D(bounds, model, y_test)
    return  wlk_inv_value


def main_wlk():
    p = QuadraticPolynomial(4, 1, 2)
    num_of_rows = 200
    if not os.path.isfile(os.pardir + f"/data/quadratic_{num_of_rows}.csv"):
        p.generate_quadratic_data(num_of_rows)
        p.save_surface()
    if not os.path.exists("mlpmodel2D"):
        X_test, model, y_test = pipeline_MLP_2D(num_of_rows)
        pickle.dump((X_test, model, y_test), open("mlpmodel2D", "wb"))
    else:
        X_test, model, y_test = pickle.load(open("mlpmodel2D", "rb"))
    bounds = (np.array(X_test.min(axis=1)), np.array(X_test.max(axis=1)))
    wlk_inv_value = inversion_wlk_2D(bounds, model, y_test)
    plot_inversion_2D(model, wlk_inv_value, X_test, y_test)
    return model, X_test, wlk_inv_value


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
            (quadratic, model, quad_X_test, quad_Y_test, quad_Z_test), open("mlpmodel3D", "wb")
        )
    else:
         quadratic, model, quad_X_test, quad_Y_test, quad_Z_test = pickle.load(
            open("mlpmodel3D", "rb")
        )
    bounds = (np.array(quad_X_test.min(axis=1)), np.array(quad_X_test.max(axis=1)))
    ga_inv_value, wlk_inv_value = inversion_3D(bounds, model, quad_Z_test)
    print(wlk_inv_value[0], np.array(quad_X_test)[0])
    print(ga_inv_value[0], np.array(quad_X_test)[0])
    plot_3D(quadratic, model, quad_X_test, quad_Y_test, quad_Z_test )
    return model, quad_X_test, wlk_inv_value, ga_inv_value


def inversion_ga_2D(bounds, model, y_test):

    ga_inv_value = invert_ga_2D(bounds, model, y_test)

    return ga_inv_value


def invert_wlk_2D(bounds, model, y_test):
    ga_inv_value, wlk_inv_value = invert_MLP_WLK_2D(bounds, model, y_test)

    return ga_inv_value, wlk_inv_value


def invert_ga_2D(bounds, model, y_test):

    ga_inv_value = invert_MLP_GA_2D(y_test, model, bounds)
    # pickle.dump(
    #     ga_inv_value, open(f"ga_inv_value_2D_{len(ga_inv_value)}", "wb")
    # )
    return ga_inv_value


def inversion_3D(bounds, model, y_test):
    if not os.path.exists("ga_inv_value_3D"):
        ga_inv_value = invert_MLP_GA_3D(y_test[0], model, bounds=bounds)
        pickle.dump((ga_inv_value, y_test[0]), open("ga_inv_value_2D", "wb"))
    else:
        ga_inv_value = pickle.load(open("ga_inv_value_3D", "rb"))

    if not os.path.exists("wlk_inv_value_3D"):
        wlk_inv_value = invert_MLP_WLK_3D(y_test[0], model, bounds=bounds)
        pickle.dump((ga_inv_value, y_test[0]), open("wlk_inv_value_3D", "wb"))
    else:
        wlk_inv_value = pickle.load(open("wlk_inv_value_3D", "rb"))
    return ga_inv_value, wlk_inv_value


def pipeline_ga_MLP_3D():
    quadratic = QuadraticPolynomial(1, 1, 2)
    num_of_rows = 400
    quad_X, quad_Y, quad_Z = quadratic.generate_quadratic_data(num_of_rows=num_of_rows)
    quadratic.plot_surface()
    X_train = np.append(quad_X, quad_Y, axis=1)
    y_train = quad_Z
    quad_X_test, quad_Y_test, quad_Z_test = quadratic.generate_quadratic_data(
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
    model.fit(X_train, y_train)
    print("Done Training")
    return quadratic, model, quad_X_test, quad_Y_test, quad_Z_test


def pipeline_MLP_2D(num_of_rows):

    df = pd.read_csv(f"data/quadratic_{num_of_rows}_2D", index_col=0)
    # df = scale_dataset(df)
    X = df["x"]
    y = df["y"]
    neuron_config = [200, 400, 300, 200]
    activation_config = ["identity", "logistic", "relu", "tanh"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    X_test, y_test = sort_2D_test_data(X_test, y_test)
    model, score = model_creation_MLP_2D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
    )
    return X_test, model, y_test


def pipeline_ga_3D():
    quadratic = QuadraticPolynomial(1, 1, 2)
    num_of_rows = 200
    quad_X, quad_Y, quad_Z = quadratic.generate_quadratic_data(num_of_rows=num_of_rows)
    quadratic.plot_surface()
    X_train = np.append(quad_X, quad_Y, axis=1)
    y_train = quad_Z
    quad_X_test, quad_Y_test, quad_Z_test = quadratic.generate_quadratic_data(
        num_of_rows=num_of_rows
    )
    # input can be only 1 dimensional. 2 sets of inputs -> add new columns
    neuron_config = [400, 600, 600, 600, 400]
    activation_config = ["selu", "selu", "sigmoid" "exponential", "linear"]
    model = model_creation_3D(X_train, activation_config, neuron_config, y_train)
    print("Done Training")
    return quadratic, model, quad_X_test, quad_Y_test, quad_Z_test


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


def sort_2D_test_data(X_test, y_test):
    df_a = pd.DataFrame(X_test, columns=["x"])
    df_a["y"] = y_test
    df_a = df_a.sort_values(by="x")
    X_test = df_a[["x"]]
    y_test = df_a["y"]
    return X_test, y_test


def scale_dataset(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=["x", "y", "z"])
    df = df.sort_values(by="x")
    return df


if __name__ == "__main__":
    # model, quad_x, wlk_inv = main_wlk()
    main_ga_2D()
    # main_ga_3D()

    #TODO - kivonni a Z tengelyből a függvényértékeket -> hibák topológiája -> átlag négyzetes hibák ->
    # TODO 2D-ben diagonális vonal  y és y kalap között mennyire diagonális
