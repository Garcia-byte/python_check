import numpy
import numpy as np
import scipy
from scipy.sparse import lil_matrix

A = np.empty(9)

# S = np.array([[4, -1, 1], [-1, 4.25, 2.75], [1, 2.75, 3.5]])
S = lil_matrix((3, 3))
S[0, 0] = 2
S[1, 1] = 3
S[2, 2] = 3
S = S.tocsc()
S = S.todense()
S = np.asarray(S)
# S = S.toarray()

V_t = np.array([[6, -0.5, 1.25]])
V = V_t.T
L = np.zeros(S.shape)
L_t = np.zeros(S.shape)
X = np.zeros(V.shape)
Y = np.zeros(V.shape)



# X = np.linalg.solve(S, V)
# print(X[1])

def solve1(n, S, V):
    L = np.zeros(S.shape)
    L_t = np.zeros(S.shape)
    for j in range(3):
        for i in range(j, 3):
            if i == j:
                # 平方差开根号，又有平方和根号
                L_t[j][j] = S[j][j]
                for k in range(j):
                    L_t[j][j] -= (L[j][k] ** 2)
                L[j][j] = np.sqrt(L_t[j][j])

            else:
                L[i][j] = (1 / L[j][j]) * (S[i][j])
                for k in range(j):
                    L[i][j] -= (1 / L[j][j]) * (L[i][k] * L[j][k])

    Y[0][0] = V[0][0] / L[0][0]
    for i in range(1, 3):
        Y[i][0] = (1 / L[i][i]) * V[i][0]
        for k in range(i):
            Y[i][0] -= (1 / L[i][i]) * (L[i][k] * Y[k][0])
    # print(Y)

    X[2][0] = Y[2][0] / L[2][2]
    for i in range(2):
        j = 1 - i
        X[j][0] = (1 / L[j][j]) * Y[j][0]
        for k in range(j + 1, 3):
            print(k)
            print(L[j][k])
            X[j][0] -= (1 / L[j][j]) * (L[k][j] * X[k][0])
    return X

X = solve1(3, S, V)
print(X[2])

# for j in range(3):
#     for i in range(j, 3):
#         if i == j:
#                 #平方和开根号?
#             L_t[j][j] = S[j][j]
#             for k in range(j):
#                 L_t[j][j] -= (L[j][k] ** 2)
#             L[j][j] = np.sqrt(L_t[j][j])
#
#         else:
#             L[i][j] = (1 / L[j][j]) * (S[i][j])
#             for k in range(j):
#                 L[i][j] -= (1 / L[j][j]) * (L[i][k] * L[j][k])
#
#
# Y[0][0] = V[0][0]/L[0][0]
# for i in range(1, 3):
#     Y[i][0] = (1/L[i][i])*V[i][0]
#     for k in range(i):
#         Y[i][0] -= (1/L[i][i])*(L[i][k]*Y[k][0])
# # print(Y)
#
# X[2][0] = Y[2][0]/L[2][2]
# for i in range(2):
#     j = 1 - i
#     X[j][0] = (1/L[j][j]) * Y[j][0]
#     for k in range(j + 1, 3):
#         print(k)
#         print(L[j][k])
#         X[j][0] -= (1/L[j][j])*(L[k][j]*X[k][0])
# print(X)










# A = np.array([1, 1, 1, 0, 0, 0])
# A = A
# B = 0
#
# print(A)
# for i in range(5):
#     B += A[i]
#
# print(B)
