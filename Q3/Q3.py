import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


class LogisticRegression:
    def __init__(self, pathX, pathY):

        self.X, self.Y = self.normalizedXY(pathX, pathY)

        print("Input data normalized for zero mean and unit variance in all dimesnsions...")
        self.m = len(self.X)
        # self.thetaList, self.costList = self.batchGradientDescent(eta, minCost, maxIter)

    def normalizedXY(self, pathX, pathY):
        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/data/q3")

        X = (pd.read_csv(pathX, header=None)).to_numpy()
        # print(X)

        X = (X - np.mean(X))/np.std(X)  # normalization

        X = np.c_[X, np.ones(len(X), np.float64)]
        # print(X)
        Y = (pd.read_csv(pathY, header= None)).to_numpy().flatten()
        # print(Y)

        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/Q3")

        return [X, Y]


    def Newton_Method(self):
        theta = np.zeros(self.X.shape[1], np.float64)  # theta initialised to zero

        distance = np.dot(self.X, theta)
        hThetaX = 1/(1 + np.exp(-distance))

        # print(distance.shape)
        # print(hThetaX.shape)

        # J_value = np.dot(self.Y, np.log(hThetaX)) + np.dot(1 - self.Y, np.log(1 - hThetaX))
        grad_J = np.dot(self.X.T, hThetaX - self.Y)

        I = np.identity(self.X.shape[0]) # identity matrix (n=1)*(n+1)
        diagonal = I * np.dot(hThetaX, (1-hThetaX)) # same size matrix but multipled by a scalar
        H = np.dot(self.X.T, np.dot(diagonal, self.X))


        self.learned_theta = theta - np.dot(np.linalg.inv(H), grad_J)
        print("The final theta learned by Newton's Method =", self.learned_theta)




    def plot_3b(self):
        sns.set()

        index_x0 = [i for i in range(self.m) if self.Y[i] == 0]
        index_x1 = [i for i in range(self.m) if self.Y[i] == 1]

        plt.figure(figsize=(6, 5))
        
        plt.scatter(self.X[index_x0][:, 1], self.X[index_x0][:, 0],
                    s=7, marker='o', c='r', label='Label=0')
        plt.scatter(self.X[index_x1][:, 1], self.X[index_x1][:, 0],
                    s=7, marker='o', c='b', label='Label=1')

        axes = plt.gca()  # get current axis
        x1_vals = np.array(axes.get_xlim())  # limit of the x axis
        # print(x_vals)
        x2_vals = -(self.learned_theta[1]*x1_vals + self.learned_theta[2])/self.learned_theta[0]
        plt.plot(x1_vals, x2_vals, c='g', linewidth=2, label='Boundary')
        
        plt.title('Logistic regression with Decision Boundary learned using\n Newton\'s method')
        plt.xlabel('Feature x1')
        plt.ylabel('Feature x2')
        plt.legend()
        plt.savefig('3b.png')
        plt.show()
        
