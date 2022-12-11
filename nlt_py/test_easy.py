import numpy
import numpy as np
from scipy.sparse import lil_matrix
import scipy.sparse


A = 4.25
B = 0.5
C = A - (B ** 2)
D = np.sqrt(C)
print(C, D)


# A = np.zeros((9, 1))
# print(A)


# A = np.empty(10)
# for i in range(10):
#     A[i] = i
# A = A.reshape(5, -1)
# print(A)
# print(A[1][2])


# A = np.mat([[1, 0], [0, 4]])
# B = lil_matrix((3, 3))
# B[0, 2] = 2
# B[1, 1] = 3
#
# C = B.toarray()
# D = B.tocsc()
# E = D.todense()
# print(C, D, E)

