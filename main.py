import logging

from experiment_utils.ga import main_ga_2_d
from function.Polynomial import Polynomial
from util.inversion_processing import process_inversion_results_2_d

logger = logging.getLogger("exp_logger")

if __name__ == "__main__":
    quadratic = Polynomial([1, 1, 2])

    num_of_rows = 10
    # model, quad_x_test, quad_y_test, quad_z_test, ga_inv_value = main_ga_3_d(function=quadratic,
    #                                                                          num_of_rows=num_of_rows)
    # process_inversion_results_3_d(model=model, x_test=quad_x_test, y_test=quad_y_test, z_test=quad_z_test,
    #                               ga_inv_value=ga_inv_value, inv_function=quadratic, num_of_rows=num_of_rows)

    function = Polynomial([1, 1, 3])
    model, X_test, Y_test, ga_inv_value = main_ga_2_d(function=function, num_of_rows=num_of_rows)
    process_inversion_results_2_d(function=function)
    # TODO - kivonni a Z tengelyből a függvényértékeket -> hibák topológiája -> átlag négyzetes hibák ->
    # TODO 2D-ben diagonális vonal  y és y kalap között mennyire diagonális
