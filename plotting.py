import numpy as np
from keras import Model
from matplotlib import pyplot as plt

from quadratic_polynomial import QuadraticPolynomial


def plot_2D(model, X_test, y_test):
    plt.plot(X_test["x"], y_test)
    prediction = model.predict(X_test)
    plt.plot(X_test["x"], prediction.flatten())
    plt.show()


def plot_3D(
    quadratic: QuadraticPolynomial, model: Model, quad_X_test, quad_Y_test, quad_Z_test
):
    # quadratic.plot_surface(quad_X_test, quad_Y_test,  quad_Z_test)
    X_test = np.append(quad_X_test, quad_Y_test, axis=1)
    quadratic.plot_surface(quad_X_test, quad_Y_test, model.predict(X_test))
    return


def plot_inversion_2D(regressor, inv_value, X, y):
    plt.plot(X["x"], y)
    # tuple((x,y) for x,y in zip(regressor.predict(np.array(inv_value)).reshape(1, -1),(-1)*regressor.predict(np.array(inv_value)).reshape(1, -1)))
    plt.scatter(
        x=(inv_value[0], -inv_value[0]),
        y=(a := regressor.predict(np.array(inv_value[0]).reshape(1, -1)), a),
    )
    if inv_value[0] < 0:
        inv_value[0] *= -1
    plt.plot(
        (inv_value[0], -X["x"].iloc[14]),
        ((regressor.predict(np.array(inv_value)[0].reshape(1, -1))), y.iloc[14]),
    )
    plt.plot(
        (-inv_value[0], X["x"].iloc[14]),
        ((regressor.predict(np.array(inv_value)[0].reshape(1, -1))), y.iloc[14]),
    )
    plt.show()
    return


def plot_inversion_3D(regressor, inv_value, X, y):
    pass
