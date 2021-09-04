import numpy as np
import pandas as pd


class LinearReg:
    def __init__(self, pathX, pathY) -> None:
        self.X, self.Y = self.parse_data(pathX, pathY)

    def parse_data(self, pathX, pathY):
        X = (pd.read_csv("data/q1/" + pathX)).to_numpy().flatten()
        Y = (pd.read_csv("data/q1/" + pathY)).to_numpy().flatten()

        return [X, Y]
