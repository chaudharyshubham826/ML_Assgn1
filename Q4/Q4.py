import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import time
import os


class GDA:
    def __init__(self, pathX, pathY):
        self.X, self.Y = self.normalizedXY(pathX, pathY)
        print("Input data normalized for zero mean and unit variance in all dimensions")



    def normalizedXY(self, pathX, pathY):
        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/data/q4")
        X = np.loadtxt(pathX) # already loads as a 2d numpy array, unlike pandas dataframe
        # print(X)

        X = (X - np.mean(X, axis= 0))/np.std(X, axis= 0)  # normalization

        # X = np.c_[X, np.ones(len(X), np.float64)]
        # print(X)
        Y = np.loadtxt(pathY, dtype= str)
        # print(Y)

        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/Q4")

        return [X, Y]


    def part_4a(self):
        m = len(self.Y)
        self.phi = np.sum([self.Y == 'Alaska'])/m # phi = Labelling Alaska as 1, and Canada = 0

        self.index_alaska = [i for i in range(len(self.Y)) if self.Y[i] == 'Alaska']
        self.index_canada = [i for i in range(len(self.Y)) if self.Y[i] == 'Canada']

        self.mean_alaska = np.mean(self.X[self.index_alaska], axis=0).reshape(-1, 1)
        self.mean_canada = np.mean(self.X[self.index_canada], axis=0).reshape(-1, 1)

        covariance = np.zeros((2, 2))

        for i in range(m):
            if(self.Y[i] == 'Alaska'):
                covariance += np.dot(self.X[i].reshape(-1, 1) - self.mean_alaska, (self.X[i].reshape(-1, 1) - self.mean_alaska).T)

            else:
                covariance += np.dot(self.X[i].reshape(-1, 1) - self.mean_canada, (self.X[i].reshape(-1, 1) - self.mean_canada).T)

        self.covariance = covariance / m

        print("Learned parameters are following: ")
        print("phi:", self.phi)
        print("mean_alaska:", self.mean_alaska)
        print("mean_canada:", self.mean_canada)
        print("Covariance matrix:")
        print(self.covariance)


    def plot_4b(self):
        sns.set()

        plt.figure(figsize=(6, 5))

        # Outline ready with the x and y values
        sns.regplot(self.X[self.index_alaska][:, 0], self.X[self.index_alaska][:, 1], fit_reg=False, marker="+", label='Alaska', color='blue')
        sns.regplot(self.X[self.index_canada][:, 0], self.X[self.index_canada][:, 1], fit_reg=False, marker=".", label='Canada', color='red')

        # plt.scatter(self.X[self.index_alaska][:, 0], self.X[self.index_alaska][:, 1],
        #             s=7, marker='+', c='blue', label='Alaska')
        # plt.scatter(self.X[self.index_canada][:, 0], self.X[self.index_canada][:, 1],
        #             s=7, marker='o', c='red', label='Canada')

        plt.title(
            'Guassian Discriminant Analysis')
        plt.xlabel('Growth ring diameter in Fresh Water')
        plt.ylabel('Growth ring diameter in Marine Water')
        plt.legend()
        plt.savefig('4b.png')
        plt.show()


    def plot_4c(self):

        sns.set()

        plt.figure(figsize=(6, 5))

        # Outline ready with the x and y values
        sns.regplot(self.X[self.index_alaska][:, 0], self.X[self.index_alaska]
                    [:, 1], fit_reg=False, marker="+", label='Alaska', color='blue')
        sns.regplot(self.X[self.index_canada][:, 0], self.X[self.index_canada]
                    [:, 1], fit_reg=False, marker=".", label='Canada', color='red')

        cov_inv = np.linalg.pinv(self.covariance)

        axes = plt.gca()

        x_vals = np.array(axes.get_xlim())

        c = np.log((1-self.phi)/self.phi)

        a = np.dot((self.mean_alaska - self.mean_canada).T, cov_inv) # 1*n

        print(a)
        
        z =  0.5*(np.dot(np.dot(self.mean_alaska.T, cov_inv), self.mean_alaska) - np.dot(np.dot(self.mean_canada.T, cov_inv), self.mean_canada))

        y_vals = (z + c - a[0][0] * x_vals)/a[0][1]

        plt.plot(x_vals, y_vals.flatten(), color='g')
        

        
        plt.title(
            'Guassian Discriminant Analysis')
        plt.xlabel('Growth ring diameter in Fresh Water')
        plt.ylabel('Growth ring diameter in Marine Water')
        plt.legend()
        plt.savefig('4c.png')
        plt.show()



    def part_4d(self):
        m = len(self.Y)

        cov_alaska = np.zeros((2, 2))
        cov_canada = np.zeros((2, 2))

        alaska_count = 0
        cananda_count = 0
        for i in range(m):
            if(self.Y[i] == 'Alaska'):
                # np.dot(self.X[i].reshape(-1, 1) - self.mean_alaska, (self.X[i].reshape(-1, 1) - self.mean_alaska).T)
                cov_alaska += np.dot(self.X[i].reshape(-1, 1) - self.mean_alaska, (self.X[i].reshape(-1, 1) - self.mean_alaska).T)
                alaska_count += 1
            else:
                cov_canada += np.dot(self.X[i].reshape(-1, 1) - self.mean_canada, (self.X[i].reshape(-1, 1) - self.mean_canada).T)
                cananda_count += 1

        self.cov_alaska = cov_alaska / alaska_count
        self.cov_canada = cov_canada / cananda_count

        print("Learned parameters are following: ")
        print("phi:", self.phi)
        print("mean_alaska:", self.mean_alaska)
        print("mean_canada:", self.mean_canada)
        print("Covariance matrix Alaska:")
        print(self.cov_alaska)
        print("Covariance matrix Canada:")
        print(self.cov_canada)


    def plot_1e(self):
        sns.set()
        plt.figure(figsize=(6, 5))

        # Outline ready with the x and y values
        sns.regplot(self.X[self.index_alaska][:, 0], self.X[self.index_alaska]
                    [:, 1], fit_reg=False, marker="+", label='Alaska', color='blue')
        sns.regplot(self.X[self.index_canada][:, 0], self.X[self.index_canada]
                    [:, 1], fit_reg=False, marker=".", label='Canada', color='red')

        cov_inv = np.linalg.pinv(self.covariance)

        axes = plt.gca()

        x_vals = np.array(axes.get_xlim())

        c = np.log((1-self.phi)/self.phi)

        a = np.dot((self.mean_alaska - self.mean_canada).T, cov_inv)  # 1*n

        # print(a)

        z = 0.5*(np.dot(np.dot(self.mean_alaska.T, cov_inv), self.mean_alaska) -
                 np.dot(np.dot(self.mean_canada.T, cov_inv), self.mean_canada))

        y_vals = (z + c - a[0][0] * x_vals)/a[0][1]

        plt.plot(x_vals, y_vals.flatten(), color='g', label= r"$\Sigma_0 = \Sigma_1$")

        covAls_inv = np.linalg.pinv(self.cov_alaska)
        covCan_inv = np.linalg.pinv(self.cov_canada)

        covAls_det = np.linalg.det(self.cov_alaska)
        covCan_det = np.linalg.det(self.cov_canada)

        b = covAls_inv - covCan_inv

        c = np.log((1-self.phi)/self.phi) + np.log(np.sqrt(covAls_det)/np.sqrt(covCan_det))


        d = 2 * (np.dot(self.mean_alaska.T, covAls_inv) - np.dot(self.mean_canada.T, covCan_inv))

        e = np.dot(np.dot(self.mean_alaska.T, covAls_inv), self.mean_alaska) - np.dot(np.dot(self.mean_canada.T, covCan_inv), self.mean_canada)

        x1_val = np.linspace(x_vals[0], x_vals[1]-0.5, 80)
        c1 = b[1, 1]
        x2_val = []
        for i in range(len(x1_val)):
            temp2 = (b[0, 0]*(x1_val[i]**2)) - (d[0, 0]*x1_val[i]) + e[0, 0]
            temp3 = ((b[0, 1]+b[1, 0])*x1_val[i]) - d[0, 1]
            x2_val.append(np.roots([c1, temp3, temp2]))

        # y_q0 = [(x2_val[i][0]) for i in range(len(x2_val))]
        y_q1 = [(x2_val[i][1]) for i in range(len(x2_val))]

        # plt.scatter(x1_val, y_q0, c='orange', s=4, label='Quad DB')
        # plt.scatter(x1_val, y_q1, c='orange', s=2,
        #             label=r"$\Sigma_0 \neq \Sigma_1$")

        sns.regplot(x1_val, y_q1, fit_reg=False, marker=".",
                    label=r"$\Sigma_0 \neq \Sigma_1$", color='black')

        plt.title(
            'Guassian Discriminant Analysis')
        plt.xlabel('Growth ring diameter in Fresh Water')
        plt.ylabel('Growth ring diameter in Marine Water')
        plt.legend()
        plt.savefig('4e.png')
        plt.show()
