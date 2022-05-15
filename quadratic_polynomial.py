import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class QuadraticPolynomial:
    def __init__(self, quadratic, linear, constant):
        self.polynomial = np.polynomial.polynomial.Polynomial(
            [constant, linear, quadratic]
        )
        print(self.polynomial.coef)

    def calculate(self, x) -> float:
        print()
        return np.polynomial.polynomial.polyval(x, self.polynomial.coef)

    def invert(self, solution):
        a = np.polynomial.Polynomial(
            np.polynomial.polynomial.polysub(self.polynomial.coef, (0, 0, solution))
        )
        return a.roots()

    def generate_quadratic_data(self, num_of_rows=1000, lower_b=-100, upper_b=100):
        range = upper_b - lower_b
        x = np.random.rand(num_of_rows, 1) * range + lower_b
        y = np.random.rand(num_of_rows, 1) * range + lower_b
        x.sort(axis=0)
        y.sort(axis=0)
        X, Y = np.meshgrid(x, y)
        Z = self.calculate(np.sqrt(X ** 2 + Y ** 2))
        self.X, self.Y, self.Z = X, Y, Z
        self.num_of_rows = num_of_rows
        return X, Y, Z

    def save_surface(
        self,
    ):
        df = pd.DataFrame(
            data={
                "x": self.X.diagonal(),
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
        if not os.path.exists("./plots"):
            os.mkdir("plots")
        plt.savefig(f"plots/quadratic_surface{datetime.now()}.pdf")
        plt.show()
