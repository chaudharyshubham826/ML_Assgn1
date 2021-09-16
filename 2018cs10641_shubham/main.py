from Q1.Q1 import LinearRegression
from Q2.Q2 import SGD
from Q3.Q3 import LogisticRegression
from Q4.Q4 import GDA

import warnings
warnings.filterwarnings("ignore")


print("Enter the question number: ")
q = int(input())

if(q == 1):
    lr = LinearRegression("linearX.csv", "linearY.csv")

    print("Running Part a...")
    lr.batchGradientDescent(0.001, 0.0001, 125)

    input("Press Enter to run Part B...")
    print("Running part B...")
    lr.plot_1b()

    input("Press Enter to run Part C...")
    print("Running part C...")
    lr.plot_1c()

    input("Press Enter to run Part D...")
    print("Running part D...")
    lr.plot_1d()

    input("Press Enter to run Part E...")
    print("Running part E...")
    lr.plot_1e()

elif(q == 2):
    sgd = SGD("q2test.csv")

    print("Running Part a...")
    sgd.sample_shuffle_2a()

    input("Press Enter to run Part b and c...")
    print("Running part b and c...")
    sgd.SGD_2b2c()

    input("Press Enter to run Part c...")
    print("Running part c...")
    sgd.part_2c()

    input("Press Enter to run Part d...")
    print("Running part d...")
    sgd.part_2d()

elif(q == 3):
    lg = LogisticRegression("logisticX.csv", "logisticY.csv")

    print("Running Part a...")
    lg.Newton_Method()

    input("Press Enter to run Part b...")
    print("Running part b...")
    lg.plot_3b()

elif(q == 4):
    gda = GDA("q4x.dat", "q4y.dat")

    input("Press Enter to run Part a...")
    print("Running part a...")
    gda.part_4a()

    input("Press Enter to run Part b...")
    print("Running part b...")
    gda.plot_4b()

    input("Press Enter to run Part c...")
    print("Running part c...")
    gda.plot_4c()

    input("Press Enter to run Part d...")
    print("Running part d...")
    gda.part_4d()

    input("Press Enter to run Part e...")
    print("Running part e...")
    gda.plot_1e()

    input("Press Enter to run Part f...")
    print("Observations mentioned in report!")


print("Done!")
# lr.plot_1c()

# lr.plot_1d()

# sgd = SGD("q2test.csv")
# sgd.SGD_2b()
# sgd.part_2d()
