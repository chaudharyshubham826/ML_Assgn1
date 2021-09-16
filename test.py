import numpy as np

a = [1, 2] # diff values for each example
c = [[3], [4]]
b = np.array([2, 4]).reshape(-1, 1)

d = np.dot(b, b.T)
print(d)
