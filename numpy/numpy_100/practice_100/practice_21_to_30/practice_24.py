# 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)

import numpy as np

# 5X3 행렬 만들기.
arr1 = np.zeros((5, 3))
print(arr1)
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

# 3X2 행렬 만들기.
arr2 = np.zeros((3, 2))
print(arr2)
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]]

# 5X3 행렬과 3X2 행렬 곱하기 연산.
result = np.dot(arr1, arr2)
print(result)
# [[0. 0.]
#  [0. 0.]
#  [0. 0.]
#  [0. 0.]
#  [0. 0.]]

# Z = np.matmul(np.ones((5, 3)), np.ones((3, 2)))
# print(Z)

# # Alternative solution, in Python 3.5 and above
# Z = np.ones((5,3)) @ np.ones((3,2))
# print(Z)