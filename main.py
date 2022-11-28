import logging

from experiment_utils.ga import main_ga_2D, main_ga_3D

logger = logging.getLogger("exp_logger")

if __name__ == "__main__":
    model_3d, quad_X_test, ga_inv_value_3D = main_ga_3D()
    model_2d, X_test, ga_inv_value_2D = main_ga_2D()
    # TODO - kivonni a Z tengelyből a függvényértékeket -> hibák topológiája -> átlag négyzetes hibák ->
    # TODO 2D-ben diagonális vonal  y és y kalap között mennyire diagonális
