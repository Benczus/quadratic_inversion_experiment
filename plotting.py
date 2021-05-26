from ast import Str

import numpy as np
from keras import Model
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor

from quadratic_polynomial import QuadraticPolynomial


def plot_2D(model, X_test, y_test):
    plt.plot(X_test['x'], y_test)
    prediction = model.predict(X_test)
    plt.plot(X_test['x'], prediction.flatten())
    plt.show()


def plot_3D(quadratic: QuadraticPolynomial, model: Model, quad_X_test, quad_Y_test, quad_Z_test):
    # quadratic.plot_surface(quad_X_test, quad_Y_test,  quad_Z_test)
    X_test = np.append(quad_X_test, quad_Y_test, axis=1)
    quadratic.plot_surface(quad_X_test, quad_Y_test, model.predict(X_test))
    return


def plot_inversion_2D(type: Str, regressor: MLPRegressor):
    pass


def plot_inversion_3D():
    pass
