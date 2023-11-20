import numpy as np
from keras.models import Model
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense


def create_default_model(neuron_config, losses=None):
    if losses is None:
        losses = ["mae"]
    model_input = Input(shape=(2))
    x = Dense(neuron_config[0], activation="linear")(model_input)
    for neurons in neuron_config[1:]:
        x = Dense(neurons, activation="linear")(x)
    model_output = Dense(1, name="output")(x)
    model = Model(model_input, outputs=model_output, name="quadratic_model")
    model.compile(optimizer="nadam", loss=losses)
    return model


def create_default_model_2D(neuron_config, activation_config, losses=None):
    if losses is None:
        losses = ["mae"]
    model_input = Input(shape=(2))
    x = Dense(neuron_config[0], activation=activation_config[0])(model_input)
    for neurons, activations in zip(neuron_config[1:], activation_config[1:]):
        x = Dense(neurons, activation=activations)(x)
    model_output = Dense(1, name="output")(x)
    model = Model(model_input, outputs=model_output, name="quadratic_model")
    model.compile(optimizer="nadam", loss=losses)
    return model


def create_default_model_3D(
        neuron_config,
        activation_config,
        input_shape,
        losses=None,
):
    if losses is None:
        losses = ["mae"]
    model_input = Input(shape=input_shape[1])
    x = Dense(neuron_config[0], activation=activation_config[0])(model_input)
    for neurons, activations in zip(neuron_config[1:], activation_config[1:]):
        x = Dense(neurons, activation="linear")(x)
    model_output = Dense(input_shape[0], name="output")(x)
    model = Model(model_input, outputs=model_output, name="quadratic_model")
    model.compile(optimizer="nadam", loss=losses)
    return model


def get_default_model_MLP_2D(activation_config, neuron_config):
    regressor = MLPRegressor(verbose=True)
    param_grid = {
        "hidden_layer_sizes": [neuron_config],
        "activation": ["relu", "logistic", "tanh"],
        "solver": ("adam",),
        "learning_rate": ["adaptive"],
    }
    cv = GridSearchCV(regressor, param_grid, verbose=True)
    return cv


def create_default_model_MLP_3D(neuron_config, activation_config):
    return get_default_model_MLP_2D(neuron_config, activation_config)


def model_creation_2D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
):
    model = create_default_model_2D(neuron_config, activation_config)
    model.fit(X_train, y_train, epochs=30)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=100)
    return model, loss_and_metrics


def model_creation_3D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
):
    model = create_default_model_3D(neuron_config, activation_config, np.shape(X_train))
    model.fit(X_train, y_train, epochs=30)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=100)
    return model, loss_and_metrics


def model_creation_MLP_2D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
):
    cv = get_default_model_MLP_2D(activation_config, neuron_config)
    cv.fit(X_train.reshape(-1, 1), y_train)
    best_reg = cv.best_estimator_
    return best_reg, best_reg.score(X_test, y_test)


def model_creation_ga_MLP_3D(
        neuron_config, activation_config, X_train, X_test, y_train, y_test
):
    cv = get_default_model_MLP_2D(activation_config, neuron_config)
    cv.fit(X_train, y_train)
    best_reg = cv.best_estimator_
    print(best_reg)
    return best_reg, best_reg.score(X_test, y_test)
