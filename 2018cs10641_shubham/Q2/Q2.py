import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
import os

class SGD:
    def __init__(self, path_test):
        self.X_test, self.Y_test = self.loadTestData(path_test)
        print("Test data loaded...")
        # self.shuffledX, self.shuffledY = self.sample_shuffle_2a()
        
        
        

    def loadTestData(self, path_test): # loading q2test.csv
        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/data/q2")

        #Importing the test data from the CSV file
        temp = (pd.read_csv(path_test)).to_numpy()
        # print(temp)

        X1_test = temp[:, 0].reshape(-1, 1)
        # print(X1_test)
        X2_test = temp[:, 1].reshape(-1, 1)
        Y_test = temp[:, 2].reshape(-1, 1).flatten()
        X0_test = np.ones((X1_test.shape))
        X_test = np.append(X0_test, X1_test, axis=1)
        X_test = np.append(X_test, X2_test, axis=1)

        os.chdir("C:/Users/Shubham/Desktop/Sem7/COL774_ML/Q2")
        
        return X_test, Y_test



    def sample_shuffle_2a(self): # preparing the sampled data and shuffling for SGD
        m = 1000000 # final count of total examples
        print("Total count of the examples:", m)
        X0 = np.ones((m, 1))  # Intercept term

        print("Preparing sample X1 ~ N(3,4) and sample X2 ~ N(-1, 4)")
        X1 = np.random.normal(3, 2, m).reshape(-1, 1)
        X2 = np.random.normal(-1, 2, m).reshape(-1, 1)

        #Make a 1Mx3 matrix
        X = np.append(X0, X1, axis=1)
        X = np.append(X, X2, axis=1)
        print("Samples for X (X0, X1, X2) generated")

        print("Preparing sample epsilon ~ N(0, 2)")
        epsilon = np.random.normal(0, np.sqrt(2), m).reshape(-1, 1).flatten()

        #Theta given in the problem statement
        self.org_theta = np.array([3, 1, 2])
        
        Y = (np.dot(X, self.org_theta) + epsilon).reshape(-1, 1)
        print("Samples for Y generated")

        #Shuffling the data
        shuffled = np.append(X, Y, axis=1) # merging the data to preserve the mapping

        np.random.shuffle(shuffled)
        X = shuffled[:, 0:3]
        Y = (shuffled[:, -1:]).flatten()

        print("Sampled data shuffled successfully...")

        print("Sampled data prepared and shuffled successfully...")

        self.shuffledX = X
        self.shuffledY = Y
        # return X, Y



    def SGD_2b2c(self):
        # self.batches = [1000000]
        self.batches = [10000, 100, 1] # 1000000, minCost = 0.000001, eta = 0.1
        # 10000, minCost = 0.0000001, eta = 0.001
        # 100, miCost = 0.0000001, eta = 0.001
        # 1, minCost = 0.01, eta = 0.001
        minCosts = [0.0000001, 0.0000001, 0.1]
        # minCosts = [0.000001]

        self.learnedTheta = [] # for comparison
        self.time_taken = [] # for comparison
        self.thetaLists = [] # for part 4
        self.num_epochs = []
        # self.errors = []

        for i in range(len(self.batches)):
            start = time.time()

            r = self.batches[i]
            minCost = minCosts[i]
            eta = 0.001 # as given in the problem statement

            #number of batches (sub batches stored in an array mini_batch)
            mini_batch = [(self.shuffledX[i:i+r, :], self.shuffledY[i:i+r]) for i in range(0, 1000000, r)]
            print("The size of one batch is = {} and the number of batches = {}".format(r, len(mini_batch)))

            theta = np.zeros(self.shuffledX.shape[1], np.float64) # theta initialized to a list of zeros
            # print(theta.shape)
            thetaList = [theta]

            costList = [] # stores the average costs
            iterations = 0

            batch_cost = []

            # for batchSize = 1 , average over every 15000 iterations
            # for batcSize  100, average over every 10000 iterations
            # for 10000, average over every 100 iterations
            # for 1000000, average over every 1 iteration
            iterToAvg = 1
            temp = 100 # helper variable for store appropriate thetas
            if(r == 1):
                iterToAvg = 15000
                temp = 100
            elif(r == 100):
                iterToAvg = 10000
                temp = 100
            elif(r == 10000):
                iterToAvg = 100
                temp = 100

            epochs = 0

            while(True):
                # theta = theta + eta*(1/m)*(sum(xi*(yi - theta*xi)))
                # theta*xi = np.dot(theta, xi)
                brk = False

                for b in mini_batch:
                    currX = b[0]
                    currY = b[1]

                    diff = currY - np.dot(currX, theta)
                    
                    theta = theta + (eta/(2*r))*(np.dot(currX.T, diff))

                    if(iterations % temp == 0):
                        thetaList.append(theta)

                    cost = (0.5/r)*np.sum((currY - np.dot(currX, theta))**2)
                    batch_cost.append(cost)
                    iterations += 1


                    if(iterations % iterToAvg == 0):
                        # iterations = 0
                        costList.append(np.mean(batch_cost))
                        # print("Current Average Cost: ", np.mean(batch_cost))
                        # print("Current theta :", theta)
                        batch_cost.clear()

                    if(len(costList) > 1 and np.fabs(costList[-1] - costList[-2]) < minCost): # stopping criteria
                        end = time.time()
                        self.learnedTheta.append(theta)
                        self.time_taken.append(end - start)
                        self.thetaLists.append(thetaList)
                        self.num_epochs.append(epochs)
                        # self.errors.append(costList[-1])
                        print("------------------------------------------------------------------------------------------------------------------------------------------")
                        print(" For Batch Size =", r, ", Time Taken:", end - start, "Theta learned: ", theta, ", Epochs:", epochs, ", Error:", costList[-1])
                        print("------------------------------------------------------------------------------------------------------------------------------------------")

                        brk = True
                        break

                epochs += 1

                if(brk):
                    break


    def part_2c(self): # prints the erros values
        self.learned_errors = []
        self.orgHyp_errors = []

        for theta in self.learnedTheta:
            cost = (0.5/len(self.Y_test))*np.sum((self.Y_test - np.dot(self.X_test, theta))**2)
            cost_org = (0.5/len(self.Y_test)) * np.sum((self.Y_test - np.dot(self.X_test, self.org_theta))**2)

            self.learned_errors.append(cost)
            self.orgHyp_errors.append(cost_org)

        for i in range(len(self.learned_errors)):
            print("For batch size:", self.batches[i])
            print("Error om original hypothesis:", self.orgHyp_errors[i])
            print("Error on learned hypothesis:", self.learned_errors[i])
            print("Difference in errors:",
                  np.fabs(self.orgHyp_errors[i] - self.learned_errors[i]))



    def part_2d(self):

        for i in range(len(self.thetaLists)):
            plt.ion()  # interactive mode on
            fig = plt.figure(figsize=(12, 8))

            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=50, azim=-72)

            ax.set_xlim(0, 3)

            ax.set_ylim(0, 1.5)
            ax.set_zlim(0, 2)

            
            ax.set_xlabel(r'$\theta_0$', labelpad=10)
            ax.set_ylabel(r'$\theta_1$', labelpad=10)
            ax.set_zlabel(r'$\theta_2$', labelpad=10)
            ax.set_title("Theta Movement until convergence, eta = 0.001\n Batch Size= {}".format(self.batches[i]), fontsize=15)

            plt.show()

            for j in range(len(self.thetaLists[i])):
                
                

                ax.plot(self.thetaLists[i][j][0], self.thetaLists[i][j][1], self.thetaLists[i][j][2],
                        linestyle='-', color='r', marker='o', markersize=2.5)

                # fig.tight_layout()
                # fig.canvas.draw()

                # plt.pause(0.02)

            plt.savefig("2d_{}.png".format(self.batches[i]))

