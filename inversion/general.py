import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ann_training import model_creation_ga_MLP_3D, model_creation_MLP_2D
from inversion.util import sort_2D_test_data
from quadratic_polynomial import QuadraticPolynomial


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
