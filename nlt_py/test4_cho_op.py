import numpy
import numpy as np


S = np.array([[4, -1, 1], [-1, 4.25, 2.75], [1, 2.75, 3.5]])
V_t = np.array([[6, -0.5, 1.25]])
V = V_t.T
X = np.zeros(V.shape)
Y = np.zeros(V.shape)

# X = np.linalg.solve(S, V)
# print(X[1])

def solve3(n, A, b):
    L = np.zeros(A.shape)
    r = n * (n-1)/2
    t = int(r)
    h = np.zeros((t, 1))
    c = np.zeros((t, 1))
    X = np.zeros(b.shape)
    Y = np.zeros(b.shape)

    L[0][0] = np.sqrt(A[0][0])

    for i in range(1, n):
        m = i * (i-1)/2
        q = int(m)
        for j in range(i):
            #c也建的不对
            c[q+j][0] = A[i][j]
            # l[i-1][j] = L[i-1][j]
            h[0][0] = c[0][0] / L[0][0]
            L[1][0] = h[0][0]
            if i > 1:
                h[q][0] = c[q][0]/L[0][0]
                L[i][0] = h[q][0]
                if j > 0:
                    h[j+q][0] = (1 / L[j][j]) * c[q+j][0]
                    for k in range(j):
                        h[j+q][0] -= (1 / L[j][j]) * (L[j][k] * h[q+k][0])
                        L[i][j] = h[q + j][0]
                        # h[j + 1][0] -= (1 / L[j][j]) * (L[j][k] * h[k这里也与i，j有关][0])
        #np.dot也不用了
        for p in range(i):
            L[i][i] -= (h[p+q][0] ** 2)
        L[i][i] = np.sqrt(A[i][i] + L[i][i])

    Y[0][0] = b[0][0] / L[0][0]
    for i in range(1, n):
        Y[i][0] = (1 / L[i][i]) * b[i][0]
        for k in range(i):
            Y[i][0] -= (1 / L[i][i]) * (L[i][k] * Y[k][0])
    # print(Y)

    X[n-1][0] = Y[n-1][0] / L[n-1][n-1]
    for i in range(n-1):
        j = (n-2) - i
        X[j][0] = (1 / L[j][j]) * Y[j][0]
        for k in range(j + 1, n):
            X[j][0] -= (1 / L[j][j]) * (L[k][j] * X[k][0])
    return X





M = solve3(3, S, V)
print(M)


    #         for k in range(i):
    #             h[i-1][0] += L_i[j][k] * c[k][0]
    #         h[i-1][0] = L[i][j]




#
#
#
# X = solve1(3, S, V)
# print(X[1])


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
