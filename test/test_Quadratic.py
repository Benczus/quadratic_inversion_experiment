import os
import unittest

import numpy as np

from quadratic_polynomial import QuadraticPolynomial


class QuadraticFunctionTests(unittest.TestCase):
    def test_quadratic_calculate(self):
        p = QuadraticPolynomial(4, 1, 2)
        self.assertEqual(p.calculate(0), 2)

    def test_quadratic_roots(self):
        p = QuadraticPolynomial(4, 1, 2)
        np.isclose(p.invert(6), [-0.78077641, 1.28077641])

    def test_generate_quadratic_data(self):
        p = QuadraticPolynomial(4, 1, 2)
        num_of_rows = 2000
        p.generate_quadratic_data(num_of_rows)
        p.save_surface()
        self.assertEqual(
            True, os.path.isfile(os.pardir + f"/data/quadratic_{num_of_rows}.csv")
        )
