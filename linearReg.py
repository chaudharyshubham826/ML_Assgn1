import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class LinearReg:
    def __init__(self, pathX, pathY, eta) -> None:
        
        self.X, self.Y = self.generateXY(pathX, pathY)
        self.dim = len(self.X[0])
        self.m = len(self.X)
        self.theta = np.zeros(self.dim, np.float64)
        self.eta = eta

    def generateXY(self, pathX, pathY):
        X = (pd.read_csv("data/q1/" + pathX)).to_numpy()
        # print(X)

        X = (X - np.mean(X))/np.std(X) #normalization

        
        X = np.c_[X, np.ones(len(X), np.float64)]
        Y = (pd.read_csv("data/q1/" + pathY)).to_numpy().flatten()

        return [X, Y]

    def batchGradientDescent(self):
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

            print(err)

            num_iter += 1

    def plot_1b(self):
        sns.set()
        sns.regplot(self.X[:, 0], self.Y, fit_reg=True, marker="+")

        axes = plt.gca()


        x_vals = np.array(axes.get_xlim())
        print(x_vals)
        y_vals = self.theta[1] + self.theta[0] * x_vals

        print(y_vals)

        plt.plot(x_vals, y_vals)


        plt.title("Linear Regression")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot_1c(self):
        pass

    def plot_1d(self):
        pass

    def plot_1e(self):
        pass