import numpy as np

a = np.array([[1], [2], [3]])

a = np.c_[a, np.ones(3, np.float64)]
b = np.array([1, 2])
c= np.array([5, 6])

d = np.dot(b.T, c)



print(np.add(b, c))