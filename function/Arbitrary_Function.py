import os
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from function.Function import Function


class Arbitrary_Function(Function):
    def __init__(self, degree):
        super().__init__(degree=degree)
        self.r1 = lambda x, y, z: x**2 + y**2 + z**2 + 8 * x * y * z
        self.r2 = lambda x, y, z: -1 * z * y

    def calculate(self, val_list: Tuple[float, float, float]) -> Tuple[float, float]:
        x, y, z = val_list
        return self.r1(x, y, z), self.r2(x, y, z)

    def invert(self, val_list):
        x, y, z = val_list
        return fsolve(self.r1, x0=x, args=(y, z)), fsolve(self.r2, x0=x, args=(y, z))

    def generate_quadratic_data_2D(self, num_of_rows=1000, lower_b=-100, upper_b=100):
        range = upper_b - lower_b
        x = np.random.rand(num_of_rows, 1) * range + lower_b
        x.sort(axis=0)
        # X, Y = np.meshgrid(x, y)
        y = self.calculate(x)
        self.x, self.y = x, y
        self.num_of_rows = num_of_rows
        return x, y

    def generate_quadratic_data_3D(self, num_of_rows=1000, lower_b=-100, upper_b=100):
        range = upper_b - lower_b
        x = np.random.rand(num_of_rows, 1) * range + lower_b
        y = np.random.rand(num_of_rows, 1) * range + lower_b
        x.sort(axis=0)
        y.sort(axis=0)
        # X, Y = np.meshgrid(x, y)
        Z = self.calculate(np.sqrt(x**2 + y**2))
        self.X, self.Y, self.Z = x, y, Z
        self.num_of_rows = num_of_rows
        return x, y, Z

    def save_surface_2D(
        self,
    ):
        df = pd.DataFrame(
            data={
                "x": self.x.diagonal(),
                "y": self.y.diagonal(),
                # "z": self.Z.diagonal(),
            }
        )
        df.to_csv(
            os.path.dirname(os.path.abspath(__file__))
            + f"/data/quadratic_{self.num_of_rows}"
        )

    def save_data_2D(
        self,
    ):
        df = pd.DataFrame(
            data={
                "x": self.x.reshape(-1),
                "y": self.y.reshape(-1),
            }
        )
        df.to_csv(
            os.path.dirname(os.path.abspath(__file__))
            + f"/data/quadratic_{self.num_of_rows}_2D"
        )

    def save_surface_3D(
        self,
    ):
        df = pd.DataFrame(
            data={
                "x": self.x.diagonal(),
                "y": self.Y.diagonal(),
                "z": self.Z.diagonal(),
            }
        )
        df.to_csv(
            os.path.dirname(os.path.abspath(__file__))
            + f"/data/quadratic_{self.num_of_rows}"
        )

    def plot_surface(self, X=None, Y=None, Z=None):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X = self.X if X is None else X
        Y = self.Y if Y is None else Y
        Z = self.Z if Z is None else Z
        print(Z)
        surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        if not os.path.exists("../plots"):
            os.mkdir("../plots")
        plt.savefig(f"plots/quadratic_surface{datetime.now()}.pdf")
        plt.show()
