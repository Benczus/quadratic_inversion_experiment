import os
import unittest

import numpy as np

from polynomial import Polynomial


class QuadraticFunctionTests(unittest.TestCase):
    def test_quadratic_calculate(self):
        p = Polynomial([4, 1, 2])
        self.assertEqual(p.calculate(0), 2)

    def test_quadratic_roots(self):
        p = Polynomial([4, 1, 2])
        np.isclose(p.invert(6), [-0.78077641, 1.28077641])

    def test_generate_quadratic_data(self):
        p = Polynomial([4, 1, 2])
        num_of_rows = 200
        p.generate_quadratic_data_2D(num_of_rows)
        p.save_surface_2D()
        self.assertEqual(
            True, os.path.isfile(os.pardir + f"/data/quadratic_{num_of_rows}.csv")
        )
