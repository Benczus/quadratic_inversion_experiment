import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from function.Function import Function


class Polynomial(Function):
    def __init__(self, values, degree=None):
        if not degree:
            degree = len(values)
        super().__init__(degree=degree)
        self.polynomial = np.polynomial.polynomial.Polynomial(values)
        print(self.polynomial.coef)

    def calculate(self, x) -> float:
        print()
        return np.polynomial.polynomial.polyval(x, self.polynomial.coef)

    def invert(self, solution):
        a = np.polynomial.Polynomial(
            np.polynomial.polynomial.polysub(self.polynomial.coef, (0, 0, solution))
        )
        return a.roots()

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
