import numpy as np
import pandas as pd


class LinearReg:
    def __init__(self, pathX, pathY, eta) -> None:
        
        self.X, self.Y = self.parse_data(pathX, pathY)
        self.dim = len(self.X[0])
        self.m = len(self.X)
        self.theta = np.zeros(self.dim, np.float64)
        self.eta = eta

    def parse_data(self, pathX, pathY):
        X = (pd.read_csv("data/q1/" + pathX)).to_numpy()
        X = np.c_[X, np.ones(len(X), np.float64)]
        Y = (pd.read_csv("data/q1/" + pathY)).to_numpy().flatten()

        return [X, Y]

    def run(self):
        # theta = np.zeros(self.dim+1, np.float64)
        num_iter = 0
        while(num_iter < 100):
            # theta = theta + eta*(1/m)*(sum(xi*(yi - theta*xi)))
            # theta*xi = np.dot(theta, xi)
            # 
            err = np.zeros(self.dim, np.float64)

            

            for i in range(self.m):

                indivErr = (self.Y[i] - np.dot(self.theta, self.X[i])) * self.X[i]

                err = np.add(err, indivErr)

            delta = (self.eta/self.m) * err
            self.theta = np.add(self.theta, delta)

            print(self.theta)

            num_iter += 1
