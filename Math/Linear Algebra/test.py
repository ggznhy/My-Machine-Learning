import numpy as np

# test 1
# A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.dtype(float))
#
# b = np.array([1, 1, 1], dtype=np.dtype(float))
#
# print(A)
# print(b)
#
# x = np.linalg.solve(A, b)
# print(x)
#
# d = np.linalg.det(A)
# print(d)


# test 2

# for j in range(1, 3):
#     print(j)


# test 3
# Y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.dtype(float))
# print(Y)
# mean_vector = np.mean(Y, axis=0)
# print(mean_vector)
# mean_matrix = np.repeat(mean_vector, repeats=Y.shape[0])
# # use np.reshape to reshape into a matrix with the same size as Y. Remember to use order='F'
# mean_matrix = np.reshape(mean_matrix, order='F', newshape=Y.shape)
# print(mean_matrix)
#
# X = Y - mean_matrix
# print(X)


# test 4
vector = np.array([[1,1],[0,1],[1,0]], dtype=np.dtype(float))
tmp = vector[0:2]
print(tmp)
V = tmp/np.linalg.norm(tmp)
print(V)