import logging

from util.inversion_processing import process_inversion_results

logger = logging.getLogger("exp_logger")

if __name__ == "__main__":
    # model_3d, quad_X_test, ga_inv_value_3D = main_ga_3D()
    # model, X_test, Y_test, ga_inv_value = main_ga_2D()
    process_inversion_results()
    # TODO - kivonni a Z tengelyből a függvényértékeket -> hibák topológiája -> átlag négyzetes hibák ->
    # TODO 2D-ben diagonális vonal  y és y kalap között mennyire diagonális
