import pickle

import numpy as np


def process_inversion_results(model=None, X_test=None, Y_test=None, ga_inv_value=None):
    if not any((model, X_test, Y_test, ga_inv_value)):
        with open("inversion_save", "rb+") as f:
            model, X_test, Y_test, ga_inv_value = pickle.load(f)
        diffs_true = []
        for y, g in zip(Y_test, ga_inv_value):
            diffs_true.append(y - model.predict(g[0]))
        print(np.mean(diffs_true))
        diffs_model = []
        for x, g in zip(X_test, ga_inv_value):
            diffs_model.append(model.predict(x.reshape(-1, 1)) - model.predict(g[0]))
        print(np.mean(diffs_model))
