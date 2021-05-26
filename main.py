import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ann_training import model_creation_3D, model_creation_2D, model_creation_MLP_2D, model_creation_MLP_3D
from inversion_util import invert_MLP_3D, invert_MLP_2D
from plotting import plot_2D, plot_3D, plot_inversion_3D, plot_inversion_2D
from quadratic_polynomial import QuadraticPolynomial

logger = logging.getLogger("exp_logger")


def main():
    X_test, model, y_test = pipeline_MLP_2D()
    plot_2D(model, X_test, y_test)
    quadratic, model, quad_X_test, quad_Y_test, quad_Z_test = pipeline_MLP_3D()
    plot_3D(quadratic, model, quad_X_test, quad_Y_test, quad_Z_test)
    # invert_MLP_2D('GA', y_test[0])
    # invert_MLP_3D('WLK', quad_Y_test[0])
    # plot_inversion_2D()
    # plot_inversion_3D()
    return model


def pipeline_MLP_3D():
    quadratic = QuadraticPolynomial(1, 1, 2)
    num_of_rows = 200
    quad_X, quad_Y, quad_Z = quadratic.generate_quadratic_data(num_of_rows=num_of_rows)
    quadratic.plot_surface()
    X_train = np.append(quad_X, quad_Y, axis=1)
    y_train = quad_Z
    quad_X_test, quad_Y_test, quad_Z_test = quadratic.generate_quadratic_data(num_of_rows=num_of_rows)
    # input can be only 1 dimensional. 2 sets of inputs -> add new columns
    X_test = np.append(quad_X_test, quad_Y_test, axis=1)
    y_test = quad_Z_test
    neuron_config = [400, 600, 600, 600, 400]
    activation_config = ['selu', 'selu', 'sigmoid' 'exponential', 'linear']
    model, score = model_creation_MLP_3D(neuron_config, activation_config, X_train, X_test, y_train, y_test)
    model.fit(X_train, y_train)
    print("Done Training")
    return quadratic, model, quad_X_test, quad_Y_test, quad_Z_test


def pipeline_MLP_2D():
    num_of_rows = 2000
    df = pd.read_csv(f'data/quadratic_{num_of_rows}', index_col=0)
    # df = scale_dataset(df)
    X = df[['x', 'y']]
    y = df['z']
    neuron_config = [200, 400, 300, 200]
    activation_config = ['identity', 'logistic', 'relu', 'tanh']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_test, y_test = sort_2D_test_data(X_test, y_test)
    model, score = model_creation_MLP_2D(neuron_config, activation_config, X_train, X_test, y_train, y_test)
    return X_test, model, y_test


def pipeline_3D():
    quadratic = QuadraticPolynomial(1, 1, 2)
    num_of_rows = 200
    quad_X, quad_Y, quad_Z = quadratic.generate_quadratic_data(num_of_rows=num_of_rows)
    quadratic.plot_surface()
    X_train = np.append(quad_X, quad_Y, axis=1)
    y_train = quad_Z
    quad_X_test, quad_Y_test, quad_Z_test = quadratic.generate_quadratic_data(num_of_rows=num_of_rows)
    # input can be only 1 dimensional. 2 sets of inputs -> add new columns
    neuron_config = [400, 600, 600, 600, 400]
    activation_config = ['selu', 'selu', 'sigmoid' 'exponential', 'linear']
    model = model_creation_3D(X_train, activation_config, neuron_config, y_train)
    print("Done Training")
    return quadratic, model, quad_X_test, quad_Y_test, quad_Z_test


def pipeline_2D():
    num_of_rows = 2000
    df = pd.read_csv(f'data/quadratic_{num_of_rows}', index_col=0)
    # df = scale_dataset(df)
    X = df[['x', 'y']]
    y = df['z']
    neuron_config = [200, 400, 300, 200]
    activation_config = ['selu', 'tanh', 'sigmoid', 'exponential']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    X_test, y_test = sort_2D_test_data(X_test, y_test)
    model, new_loss = model_creation_2D(neuron_config, activation_config, X_train, X_test, y_train, y_test)
    return X_test, model, y_test


def sort_2D_test_data(X_test, y_test):
    df_a = pd.DataFrame(X_test, columns=['x', 'y'])
    df_a['z'] = y_test
    df_a = df_a.sort_values(by='x')
    X_test = df_a[['x', 'y']]
    y_test = df_a['z']
    return X_test, y_test


def scale_dataset(df):
    scaler = StandardScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=['x', 'y', 'z'])
    df = df.sort_values(by='x')
    return df


if __name__ == "__main__":
    model = main()
