import logging

from experiment_utils.ga import main_ga_2D
from function.Polynomial import Polynomial
from util.inversion_processing import process_inversion_results

logger = logging.getLogger("exp_logger")

if __name__ == "__main__":
    # model_3d, quad_X_test, ga_inv_value_3D = main_ga_3D()
    function = Polynomial([1, 1, 3])
    model, X_test, Y_test, ga_inv_value = main_ga_2D(function=function)
    process_inversion_results(function=function)
    # TODO - kivonni a Z tengelyből a függvényértékeket -> hibák topológiája -> átlag négyzetes hibák ->
    # TODO 2D-ben diagonális vonal  y és y kalap között mennyire diagonális
