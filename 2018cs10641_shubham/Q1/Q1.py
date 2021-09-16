import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import time
import os
# from mpl_toolkits.mplot3d import Axes3D


class LinearRegression:
    def __init__(self, pathX, pathY):
        
        self.X, self.Y = self.normalizedXY(pathX, pathY)

        print("Input data normalized for zero mean and unit variance")
        self.m = len(self.X)
        # self.thetaList, self.costList = self.batchGradientDescent(eta, minCost, maxIter)

    def normalizedXY(self, pathX, pathY):
        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/data/q1")
        X = (pd.read_csv(pathX)).to_numpy()
        # print(X)

        X = (X - np.mean(X))/np.std(X) #normalization
        
        X = np.c_[X, np.ones(len(X), np.float64)]
        # print(X)
        Y = (pd.read_csv(pathY)).to_numpy().flatten()

        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/Q1")

        return [X, Y]



    def batchGradientDescent(self, eta, minCost, maxIter): #BATCH GRADIENT DESCENT IMPLEMENTATION
        self.eta = eta

        theta = np.zeros(self.X.shape[1], np.float64) # theta initialised to zero
        thetaList = [theta]
        cost = 0.5*np.sum((self.Y - np.dot(self.X, theta))**2) #initial cost
        costList = [cost]
        iterations = 0

        while(iterations < maxIter and cost > minCost): #stopping criteria
            # theta = theta + eta*(1/m)*(sum(xi*(yi - theta*xi)))
            # theta*xi = np.dot(theta, xi)
            diff = self.Y - np.dot(self.X, theta)
            theta = theta + (eta)*(np.dot(self.X.T, diff))
            
            thetaList.append(theta)

            cost = 0.5*np.sum((self.Y - np.dot(self.X, theta))**2)
            costList.append(cost)
            iterations += 1
            print(iterations, "-", cost)

        print("Cost -", cost)
        print("Theta -", theta)
        self.thetaList = thetaList
        self.costList = costList




    def plot_1b(self):
        sns.set()
        # print(self.X[:, 0])
        # print(self.Y)
        sns.regplot(self.X[:, 0], self.Y, fit_reg=False, marker="+") # Outline ready with the x and y values
        axes = plt.gca() # get current axis
        x_vals = np.array(axes.get_xlim()) # limit of the x axis
        # print(x_vals)
        y_vals = self.thetaList[-1][1] + self.thetaList[-1][0] * x_vals

        # print(y_vals)

        plt.plot(x_vals, y_vals)

        plt.title("Linear Regression")
        plt.xlabel("Acidity of wine(Normalised)")
        plt.ylabel("Density of wine")
        plt.savefig("1b.png")
        plt.show()



    def plot_1c(self):
        sns.set() # resets the RC params to original settings
        # Create a mesh using numpy mgrid
        T0, T1 = np.mgrid[0:2:50j, -1:1:50j]
        mesh = np.c_[T1.flatten(), T0.flatten()]

        # Compute Cost values for the grid
        costValues = (
            np.array([self.cost(point) for point in mesh])
            .reshape(50, 50)
        )

        plt.ion() # interactive mode on

        fig = plt.figure(figsize=(7, 7))

        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=50, azim=-72) # initial viewing angle

        ax.plot_surface(T0, T1, costValues, cmap=cm.RdBu_r)

        ax.set_xlabel(r'$\theta_0$', labelpad=10)
        ax.set_ylabel(r'$\theta_1$', labelpad=10)
        ax.set_zlabel("Cost", labelpad=10)
        ax.set_title("Linear Regression - Gradient Descent")

        plt.show()

        for j in range(len(self.thetaList)):

            ax.plot(self.thetaList[j][1], self.thetaList[j][0], self.costList[j], linestyle='-', color='orange', marker='o', markersize = 2.5)

            # fig.tight_layout()
            # fig.canvas.draw()

            plt.pause(0.2)


        # save(plt, "q1_a.png")
        plt.savefig("1c.png")


    def plot_1d(self, imageName= None, mesh_limit1 = None, mesh_limit2 = None, numContours = None): # Contour plot
        sns.set()
        meshLim1 = 0 if mesh_limit1 == None else mesh_limit1
        meshLim2 = 2 if mesh_limit2 == None else mesh_limit2
        T0, T1 = np.mgrid[meshLim1:meshLim2:50j, -1:1:50j]
        mesh = np.c_[T1.flatten(), T0.flatten()]

        # Compute Cost values for the grid
        costValues = (
            np.array([self.cost(point) for point in mesh]).reshape(T0.shape)
        )

        plt.ion()  # interactive mode on

        # fig = plt.figure(figsize=(7, 7))
        numCont = 25 if numContours == None else numContours

        plt.contour(T0, T1, costValues, numCont, colors= "k")

        plt.xlim(mesh_limit1, mesh_limit2)
        

        plt.xlabel(r'$\theta_0$', labelpad=5)
        plt.ylabel(r'$\theta_1$', labelpad=5)
        plt.title("Contours, for eta = {}".format(self.eta))

        plt.show()

        for j in range(len(self.thetaList)):

            plt.plot(self.thetaList[j][1], self.thetaList[j][0], linestyle='-', color='orange', marker='o', markersize=2.5)

            # fig.tight_layout()
            # fig.canvas.draw()

            plt.pause(0.2)

        # save(plt, "q1_a.png")
        if(imageName == None):
            plt.savefig("1d.png")
        else:
            plt.savefig(imageName)



    def plot_1e(self):
        self.batchGradientDescent(0.018, 0.0001, 50)
        self.plot_1d(imageName="1e_0.018.png")

        input('Press Enter to generate the next plot...')

        self.batchGradientDescent(0.025, 0.0001, 10)
        self.plot_1d(imageName="1e_0.025.png", mesh_limit1=-
                     50, mesh_limit2=50, numContours=100)

    def cost(self, theta):
        """Cost function for linear regression."""
        return 0.5*np.sum((self.Y - np.dot(self.X, theta))**2)
