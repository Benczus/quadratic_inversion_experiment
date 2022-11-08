import logging
import pickle

from inversion_util import (invert_MLP_GA_2D, invert_MLP_GA_3D,
                            invert_MLP_WLK_2D, invert_MLP_WLK_3D)

logger = logging.getLogger("exp_logger")


def inversion_3D(bounds, model, y_test, method="ga", save=False, load=False):
    inv_value = None
    if not load:
        if method.lower() == "ga":
            inv_value = invert_MLP_GA_3D(y_test[0], model, bounds=bounds)
        if method.lower() == "wlk":
            inv_value = invert_MLP_WLK_3D(y_test[0], model, bounds=bounds)
        if save:
            pickle.dump((inv_value, y_test[0]), open(f"{method}_inv_value_2D", "wb"))
    else:
        inv_value = pickle.load(open("f{method}_inv_value_3D", "rb"))
    return inv_value


def inversion_2D(bounds, model, y_test, method="ga", save=False, load=False):
    inv_value = None
    if not load:
        if method.lower() == "ga":
            inv_value = invert_MLP_GA_2D(y_test[0], model, bounds=bounds)
        if method.lower() == "wlk":
            inv_value = invert_MLP_WLK_2D(y_test[0], model, bounds=bounds)
        if save:
            pickle.dump((inv_value, y_test[0]), open(f"{method}_inv_value_2D", "wb"))
    else:
        inv_value = pickle.load(open("f{method}_inv_value_3D", "rb"))
    return inv_value


if __name__ == "__main__":
    model, quad_x, wlk_inv = main_wlk_2D()
    model, X_test, ga_inv_value = main_ga_2D()
    main_ga_3D()
    # TODO - kivonni a Z tengelyből a függvényértékeket -> hibák topológiája -> átlag négyzetes hibák ->
    # TODO 2D-ben diagonális vonal  y és y kalap között mennyire diagonális
